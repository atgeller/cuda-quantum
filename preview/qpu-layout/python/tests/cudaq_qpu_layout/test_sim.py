# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Invariants the layout simulator must uphold, plus a few worked examples."""

import os
import pytest

from cudaq_qpu_layout.model import QpuModel
from cudaq_qpu_layout.sim import simulate, LayoutError
from cudaq_qpu_layout.trace import (replay, COMPUTE, IN, OUT, CROSS,
                                    PORT_IN, PORT_OUT)

PAYLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "payloads")


def run(name, **kwargs):
    with open(os.path.join(PAYLOADS, name)) as f:
        return simulate(f.read(), QpuModel(**kwargs)).to_json()


# ===----------------------------------------------------------------------=== #
# Invariants
# ===----------------------------------------------------------------------=== #


# Payloads are plain wire-form: the model lowers them onto its own regions, so
# the QPU geometry is the model's to vary. Sizes have slack because
# `assign-subcircuit-regions` is round-robin and not capacity-aware, so an
# assignment that exactly fits the qubit count is generally infeasible.
@pytest.fixture(params=[
    ("bell_cross.mlir", dict(num_regions=2, region_size=4)),
    ("bell_cross.mlir", dict(num_regions=2, region_size=6)),
    ("bell_cross.mlir", dict(num_regions=1, region_size=4)),
    ("line_chain.mlir", dict(num_regions=2, region_size=4)),
    ("line_chain.mlir", dict(num_regions=1, region_size=4)),
    # A fresh qubit entering a gate whose other operand is busy: the case that
    # once emitted a qubit's placement after the moves that carried it.
    ("assign_after_move.mlir", dict(num_regions=2, region_size=2)),
])
def trace(request):
    name, model = request.param
    return run(name, **model)


def test_operands_are_colocated(trace):
    """A gate can only run on qubits sharing one region."""
    for step in trace["steps"]:
        for op in step["ops"]:
            regions = {q["region"] for q in op["controls"] + op["targets"]}
            assert regions == {op["region"]}, \
                f"op {op['gate']} at t={step['t']} spans regions {regions}"


def test_deltas_are_causally_ordered(trace):
    """The delta stream must replay from an empty QPU.

    A qubit is placed before it is moved, and a move starts where the qubit
    actually is. These held inside the simulator but not in what it emitted:
    `assign` was stamped with the operation's ready time (its slowest operand)
    while the routing legs were stamped with the qubit's own, so a qubit that
    was idle while its partner was busy had its placement recorded after its
    own moves.
    """
    live = {}
    for step in trace["steps"]:
        for d in step["deltas"]:
            vq, event = d["vq"], d["event"]
            if event == "release":
                live.pop(vq, None)
                continue
            if event == "move":
                assert vq in live, \
                    f"q{vq} moved at t={step['t']} before being placed"
                src = d["from"]
                assert live[vq] == (src["region"], src["kind"]), \
                    (f"q{vq} moved from {src} at t={step['t']} but was at "
                     f"{live[vq]}")
            elif event == "assign":
                assert vq not in live, \
                    f"q{vq} re-placed at t={step['t']} while already live"
            dst = d["to"]
            live[vq] = (dst["region"], dst["kind"])


def test_capacity_is_never_exceeded(trace):
    """Compute wires are the scarce resource; ports are unbounded staging."""
    size = trace["model"]["region_size"]
    for t, pos in replay(trace):
        counts = {}
        for region, kind in pos.values():
            if kind == COMPUTE:
                counts[region] = counts.get(region, 0) + 1
        for region, n in counts.items():
            assert n <= size, f"region {region} holds {n} > {size} at t={t}"


def test_ops_only_run_on_compute_wires(trace):
    for (t, pos), step in zip(replay(trace), trace["steps"]):
        for op in step["ops"]:
            for q in op["controls"] + op["targets"]:
                assert pos[q["vq"]][1] == COMPUTE, \
                    f"op {op['gate']} at t={t} uses vq {q['vq']} on a port"


def test_every_crossing_is_bracketed_by_ports(trace):
    """A region is entered and left only through ports, so each qubit's move
    sequence must read port-out, cross, port-in."""
    per_qubit = {}
    for step in trace["steps"]:
        for m in step["moves"]:
            per_qubit.setdefault(m["vq"], []).append(m)
    for vq, moves in per_qubit.items():
        for i, m in enumerate(moves):
            if m["kind"] != CROSS:
                continue
            assert m["from"]["kind"] == OUT and m["to"]["kind"] == IN, m
            assert i > 0 and moves[i - 1]["kind"] == PORT_OUT, \
                f"vq {vq} crossed without first reaching an out-port"
            assert i + 1 < len(moves) and moves[i + 1]["kind"] == PORT_IN, \
                f"vq {vq} crossed but never landed on a compute wire"


def test_port_hops_come_in_pairs(trace):
    summary = trace["summary"]["moves"]
    assert summary["port"] == 2 * summary["cross"], summary


def test_no_qubit_is_released_from_a_port(trace):
    """A port is transit, not storage: a qubit ends its life on a compute wire."""
    for step in trace["steps"]:
        for d in step["deltas"]:
            if d["event"] == "release":
                assert d["from"]["kind"] == COMPUTE, \
                    f"vq {d['vq']} was released while parked on a port"


def test_ops_reference_the_replayed_placement(trace):
    """Each op's recorded slots agree with the state the deltas reconstruct."""
    for (t, pos), step in zip(replay(trace), trace["steps"]):
        for op in step["ops"]:
            for q in op["controls"] + op["targets"]:
                assert pos[q["vq"]] == (q["region"], COMPUTE), \
                    f"op {op['gate']} at t={t} disagrees on vq {q['vq']}"


def test_summary_matches_steps(trace):
    moves = [m for s in trace["steps"] for m in s["moves"]]
    summary = trace["summary"]
    assert summary["moves"]["cross"] == sum(1 for m in moves
                                            if m["kind"] == CROSS)
    assert summary["moves"]["port"] == sum(
        1 for m in moves if m["kind"] in (PORT_IN, PORT_OUT))
    assert summary["total_move_ticks"] == sum(m["cost"] for m in moves)
    op_steps = [s["t"] for s in trace["steps"] if s["ops"]]
    assert summary["depth"] == (max(op_steps) + 1 if op_steps else 0)


def test_moves_are_between_distinct_sites(trace):
    for step in trace["steps"]:
        for m in step["moves"]:
            assert m["from"] != m["to"]
            src, dst = m["from"], m["to"]
            if m["kind"] == PORT_OUT:
                assert src["region"] == dst["region"]
                assert (src["kind"], dst["kind"]) == (COMPUTE, OUT)
            elif m["kind"] == PORT_IN:
                assert src["region"] == dst["region"]
                assert (src["kind"], dst["kind"]) == (IN, COMPUTE)
            else:
                assert m["kind"] == CROSS
                assert (src["kind"], dst["kind"]) == (OUT, IN)


# ===----------------------------------------------------------------------=== #
# Worked examples
# ===----------------------------------------------------------------------=== #


def test_independent_ops_share_a_timestep():
    """Scheduling is still the model's job, so independent gates run together.

    Placement comes from the passes now, but nothing tells the model *when* to
    run anything -- the ASAP scheduler does, and two gates on disjoint qubits
    have no reason to wait for each other.
    """
    trace = run("bell_cross.mlir", num_regions=2, region_size=3)
    widest = max(len(s["ops"]) for s in trace["steps"])
    assert widest > 1, "every gate was serialized"


def test_crossings_are_what_the_ir_asked_for():
    """Movement is the passes' decision, costed out rather than chosen here.

    The cross-pair interaction spans two regions, so `add-region-moves` emits a
    transfer for it; the model turns each into the three legs a region crossing
    takes and charges a tick apiece.
    """
    trace = run("bell_cross.mlir", num_regions=2, region_size=3)
    assert trace["summary"]["moves"] == {"cross": 3, "port": 6}
    # Every leg is one tick.
    assert trace["summary"]["total_move_ticks"] == \
        trace["summary"]["moves"]["cross"] + trace["summary"]["moves"]["port"]

    # Every transferred qubit lands on a compute wire, not stranded on a port.
    # Check the last step that still has anyone placed: by the true final step
    # every qubit has been released, so an assertion there is vacuous.
    live = [pos for _, pos in replay(trace) if pos]
    assert live, "trace never placed a qubit"
    assert all(kind == COMPUTE for _, kind in live[-1].values()), live[-1]


def test_one_big_region_needs_no_movement():
    """With room for everything in one region, nothing ever moves."""
    trace = run("bell_cross.mlir", num_regions=1, region_size=4)
    assert trace["summary"]["total_move_ticks"] == 0


def test_co_located_qubits_never_move():
    """Within one subcircuit a region is all-to-all: gates there cost no movement.

    Movement is charged at subcircuit boundaries, not between gates, so a
    subcircuit's own residents interact with nothing in between. Crossing a
    boundary is a port round-trip even when the qubit returns to the region it
    just left -- the boundary is what costs, not the distance.
    """
    trace = run("line_chain.mlir", num_regions=1, region_size=4)
    for step in trace["steps"]:
        movers = {m["vq"] for m in step["moves"]}
        acting = {q["vq"] for op in step["ops"]
                  for q in op["controls"] + op["targets"]}
        assert not (movers & acting), \
            f"a qubit moved and ran a gate in the same step: {movers & acting}"


def test_measurements_are_counted():
    trace = run("line_chain.mlir", num_regions=1, region_size=4)
    assert trace["summary"]["num_measurements"] == 1


def test_control_flow_is_rejected():
    with pytest.raises(LayoutError, match="straight-line"):
        run("has_loop.mlir")


def test_over_capacity_is_reported():
    """A region assignment that does not fit is reported, not repaired.

    Lowering a circuit onto a model caps a subcircuit at `region_size` wires,
    so the passes fit by construction and this cannot be reached that way. It
    is reachable with IR that already carries its own region assignment, which
    is the case the guard exists for -- there is no placer left to repair it.
    """
    src = """
    quake.wire_set @wires[2]
    quake.region @r0[1]
    func.func @over() attributes {"cudaq-entrypoint"} {
      %0 = quake.borrow_wire @wires[0] : !quake.wire
      %1 = quake.borrow_wire @wires[1] : !quake.wire
      %2:2 = quake.x [%0] %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      quake.return_wire %2#0 : !quake.wire
      quake.return_wire %2#1 : !quake.wire
      return
    }
    """
    with pytest.raises(LayoutError, match="full"):
        simulate(src, QpuModel(num_regions=1, region_size=1))


def test_multi_qubit_gate_is_supported():
    """An all-to-all region places no arity limit on an operation."""
    src = """
    quake.wire_set @wires[3]
    func.func @toffoli() attributes {"cudaq-entrypoint"} {
      %q0 = quake.borrow_wire @wires[0] : !quake.wire
      %q1 = quake.borrow_wire @wires[1] : !quake.wire
      %q2 = quake.borrow_wire @wires[2] : !quake.wire
      %r:3 = quake.x [%q0, %q1] %q2
          : (!quake.wire, !quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire, !quake.wire)
      quake.return_wire %r#0 : !quake.wire
      quake.return_wire %r#1 : !quake.wire
      quake.return_wire %r#2 : !quake.wire
      return
    }
    """
    builder = simulate(src, QpuModel(num_regions=1, region_size=3))
    assert builder.to_json()["summary"]["total_move_ticks"] == 0
