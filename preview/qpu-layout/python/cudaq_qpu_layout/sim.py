#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reference model for laying a virtual circuit out on a region-based QPU.

Walks value-semantics (wire-set) Quake IR and *makes its own* layout decisions:
where each virtual qubit lives, how qubits move to meet, and which timestep each
operation runs in. It does not simulate quantum state -- the output is a
`trace.py` document describing the placement and schedule it chose.

Because the model decides everything itself, it consumes plain virtual-qubit
Quake and deliberately ignores any `{region = @rN}` / `quake.move` the
region-lowering passes may have added: those passes are an alternative path to
compare against, not an input.

Movement follows the calling convention of `LoweringQubitsToRegions.md`: a
region is entered and left only through its ports, so carrying a qubit from one
region to another is three legs -- compute wire onto an out-port, out-port
across to the destination's in-port, in-port onto a compute wire. Displacing a
resident to make room is the same act: it leaves through an out-port and, being
still live, must be carried to a compute wire elsewhere.

Placement, routing and scheduling are separate policy objects so a better
heuristic can replace one without disturbing the others -- the contract
described in `QubitLayout.md`.

Not modeled yet:
  - Pinnacle Clifford-frame merging, which serializes non-Clifford operations
    across regions that have been entangled together.
  - Heterogeneous region sizes and magic-state factories.
  - Classically-conditioned operations; a measurement is recorded as an ordinary
    op and its outcome is never used.

Run standalone:
    python3 layout_sim.py payload.mlir --regions 2 --region-size 4
"""

from cudaq.mlir.ir import Module
from cudaq.kernel.utils import getMLIRContext

from .model import QpuModel
from .trace import (TraceBuilder, COMPUTE, IN, OUT, PORT_IN, PORT_OUT,
                    CROSS)

# Every operation -- a gate, or one leg of a move -- takes one tick.
TICK = 1

# Which partitioner the lowering uses. `greedy-pair` forms partitions from
# interactions alone; `greedy` weighs every op.
import os as _os
PARTITIONER = _os.environ.get("QPU_LAYOUT_PARTITIONER", "greedy")

WIRE_TYPE = "!quake.wire"
WIRE_SOURCES = ("quake.borrow_wire", "quake.null_wire")
WIRE_SINKS = ("quake.return_wire", "quake.sink")
MEASUREMENTS = ("quake.mz", "quake.my", "quake.mx")
IGNORED = ("func.return", "cc.return", "quake.discriminate", "quake.wire_set",
           "quake.region")

# The region-lowering passes' output, which is what dictates placement:
# `outline-partitions` makes each subcircuit a lambda, `assign-subcircuit-regions`
# tags it `{region = @rN}`, and `add-region-moves` inserts the transfers.
SUBCIRCUIT = "cc.create_lambda"
SUBCIRCUIT_CALL = "cc.call_callable"
MOVE = "quake.move"
REGION_ATTR = "region"


def _region_index(symbol):
    """`@r3` -> 3. The passes name regions positionally."""
    name = str(symbol).lstrip("@")
    if not name.startswith("r") or not name[1:].isdigit():
        raise LayoutError(f"region symbol '{symbol}' is not of the form @rN")
    return int(name[1:])


def _attr(op, name):
    try:
        return op.attributes[name]
    except KeyError:
        return None


class LayoutError(RuntimeError):
    """The circuit cannot be laid out on the given QPU model."""


# ===----------------------------------------------------------------------=== #
# Placement state
# ===----------------------------------------------------------------------=== #


class Placement:
    """Which virtual qubit is in which region, and whether it is on a port.

    A site is `(region, kind)`. Wires inside a region are deliberately not
    identified: the region is all-to-all, so which one a qubit sits on is not a
    question the model can answer, and answering it anyway would be the model
    making a placement decision the passes are supposed to own. Only the count
    is real, and it is what `region_size` bounds.

    What is still tracked per wire is when each comes free, since a qubit cannot
    land on a wire its predecessor has not left yet; `free_times` holds those
    instants without naming the wires they belong to.
    """

    def __init__(self, model):
        self.model = model
        self.pos = {}  # vq -> (region, kind)
        self.occupants = {}  # (region, kind) -> {vq}
        self.free_times = {}  # region -> one entry per unoccupied compute wire
        for region in range(model.num_regions):
            for kind in (COMPUTE, IN, OUT):
                self.occupants[(region, kind)] = set()
            self.free_times[region] = [0] * model.region_size

    def region_of(self, vq):
        return self.pos[vq][0]

    def kind_of(self, vq):
        return self.pos[vq][1]

    def has_room(self, region):
        return len(self.occupants[(region, COMPUTE)]) < self.model.region_size

    def residents(self, region):
        return list(self.occupants[(region, COMPUTE)])

    def occupancy(self, region):
        return len(self.occupants[(region, COMPUTE)])

    def available_at(self, region, kind):
        """The earliest a qubit may land here.

        Ports are unbounded, so one is always free. For a compute wire this is
        whichever has been free longest -- the model does not care which.
        """
        if kind != COMPUTE:
            return 0
        free = self.free_times[region]
        return min(free) if free else 0

    def _take_wire(self, region, kind):
        if kind != COMPUTE:
            return
        free = self.free_times[region]
        if not free:
            raise LayoutError(
                f"r{region} has no free compute wire "
                f"(region_size={self.model.region_size})")
        free.remove(min(free))

    def _release_wire(self, region, kind, vacated_at):
        if kind == COMPUTE:
            self.free_times[region].append(vacated_at)

    def assign(self, vq, region, kind):
        self._take_wire(region, kind)
        self.pos[vq] = (region, kind)
        self.occupants[(region, kind)].add(vq)

    def relocate(self, vq, region, kind, vacated_at=0):
        old = self.pos[vq]
        self.occupants[old].discard(vq)
        self._release_wire(old[0], old[1], vacated_at)
        self.assign(vq, region, kind)
        return old

    def release(self, vq, vacated_at=0):
        old = self.pos.pop(vq)
        self.occupants[old].discard(vq)
        self._release_wire(old[0], old[1], vacated_at)
        return old


class Mover:
    """Costs out the moves the IR asks for. It decides nothing.

    `add-region-moves` emits the three legs a region crossing takes -- off the
    compute wire onto an out-port, across to the destination's in-port, then
    onto a compute wire there -- so each `quake.move` is one leg. This turns
    each into a tick and a trace entry; where a qubit goes, and by which route,
    was settled by the passes.
    """

    KINDS = {OUT: PORT_OUT, IN: CROSS, COMPUTE: PORT_IN}

    def __init__(self, model, placement, scheduler, builder):
        self.model = model
        self.placement = placement
        self.sched = scheduler
        self.builder = builder

    def move(self, vq, region, kind):
        """Execute one leg of a move: `quake.move %w to @rN [in|out]`."""
        if kind == COMPUTE and not self.placement.has_room(region):
            raise LayoutError(
                f"the IR moves virtual qubit {vq} onto a compute wire of "
                f"r{region}, which is full "
                f"(region_size={self.model.region_size})")
        # Scheduling is the one thing still decided here: a leg runs as soon as
        # the qubit is free and the site it lands on has been vacated.
        t = max(self.sched.time_for([vq]),
                self.placement.available_at(region, kind))
        src = self.placement.relocate(vq, region, kind, vacated_at=t + TICK)
        self.builder.move(t, vq, self.KINDS[kind], src, (region, kind), TICK)
        self.sched.commit([vq], t, TICK)
        self.builder.observe_occupancy(region, self.placement.occupancy(region))


class AsapScheduler:
    """As-soon-as-possible list scheduling.

    Each virtual qubit carries the timestep at which it is next free. An
    operation runs as soon as all of its qubits are free, so operations on
    disjoint qubits -- in one region or across regions -- share a timestep.
    That parallelism is what the trace exists to show.
    """

    def __init__(self):
        self.ready = {}

    def ready_time(self, vq):
        return self.ready.get(vq, 0)

    def time_for(self, vqs):
        return max((self.ready_time(vq) for vq in vqs), default=0)

    def commit(self, vqs, t, cost):
        for vq in vqs:
            self.ready[vq] = t + cost


# ===----------------------------------------------------------------------=== #
# IR walking
# ===----------------------------------------------------------------------=== #


def _children(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield inner.operation


def _is_wire(value):
    return str(value.type) == WIRE_TYPE


def _find_entrypoint(module):
    for op in _children(module.operation):
        if op.name != "func.func":
            continue
        for attr in op.attributes:
            if attr == "cudaq-entrypoint":
                return op
    return None


def _check_straight_line(func):
    """Reject anything the model cannot cost: branches and unrolled-away loops.

    A subcircuit lambda is the one nested region allowed through: its body is
    straight-line code that runs where the `cc.call_callable` says it does, so
    the walker inlines it at the call rather than treating it as control flow.
    """
    body = func.regions[0]
    if len(body.blocks) > 1:
        raise LayoutError(
            f"kernel '{func.opview.sym_name.value}' has "
            f"{len(body.blocks)} basic blocks; the layout simulator handles "
            "straight-line kernels only. Lower with a pipeline that fully "
            "unrolls loops and flattens branches.")
    for op in _children(func):
        if op.regions and op.name != SUBCIRCUIT:
            raise LayoutError(
                f"'{op.name}' carries a nested region; the layout simulator "
                "handles straight-line kernels only. Lower with a pipeline "
                "that fully unrolls loops and flattens branches.")


def _constant_of(value):
    """The literal value of a gate parameter, when it is a constant."""
    try:
        owner = value.owner.opview
    except Exception:
        return None
    for name in ("arith.constant", "complex.constant"):
        if value.owner.name == name:
            try:
                return owner.value.value
            except Exception:
                return None
    return None


class Simulator:

    def __init__(self, model):
        self.model = model
        self.builder = TraceBuilder(model)
        self.placement = Placement(model)
        self.sched = AsapScheduler()
        self.mover = Mover(model, self.placement, self.sched, self.builder)
        self.vq_of = {}  # MLIR wire Value -> virtual qubit id
        self._next_vq = 0
        self.subcircuits = {}  # callable Value -> its cc.create_lambda op
        self.region = 0  # the region whose subcircuit is executing

    # -- virtual qubit identity, recovered from the SSA data flow -------------

    def _new_vq(self, value):
        vq = self._next_vq
        self._next_vq += 1
        self.vq_of[value] = vq
        self.builder.num_vqubits += 1
        return vq

    def _vq(self, value):
        if value not in self.vq_of:
            return self._new_vq(value)
        return self.vq_of[value]

    def _thread(self, op):
        """Carry virtual qubit ids from `!quake.wire` operands to results."""
        ins = [o for o in op.operands if _is_wire(o)]
        outs = [r for r in op.results if _is_wire(r)]
        for src, dst in zip(ins, outs):
            self.vq_of[dst] = self._vq(src)

    # -- walking -------------------------------------------------------------

    def run(self, module):
        entry = _find_entrypoint(module)
        if entry is None:
            raise LayoutError("no `cudaq-entrypoint` function found in payload")
        _check_straight_line(entry)
        self.builder.entry = entry.opview.sym_name.value
        for op in _children(entry):
            self._visit(op)
        return self.builder

    def _visit(self, op):
        name = op.name

        if name == SUBCIRCUIT:
            # The body runs at the call, in the region this lambda is tagged
            # with; remember it until then.
            self.subcircuits[op.results[0]] = op
            return

        if name == SUBCIRCUIT_CALL:
            self._call_subcircuit(op)
            return

        if name == MOVE:
            self._move(op)
            return

        if name in WIRE_SOURCES:
            # A qubit born inside a subcircuit belongs to that subcircuit's
            # region, which is all the IR says and all the model needs.
            self._place(self._new_vq(op.results[0]), self.region)
            return

        if name in WIRE_SINKS:
            vq = self._vq(op.operands[0])
            if vq in self.placement.pos:
                t = self.sched.ready_time(vq)
                self.builder.delta(t, "release", vq,
                                   src=self.placement.release(vq, vacated_at=t))
            return

        if name in IGNORED:
            return

        view = op.opview
        if not hasattr(view, "targets"):
            self.builder.unmodeled.append(name)
            return

        self._gate(op, view)

    def _call_subcircuit(self, op):
        """Run a subcircuit's body in the region the passes assigned it."""
        lambda_op = self.subcircuits.get(op.operands[0])
        if lambda_op is None:
            raise LayoutError(
                "cc.call_callable does not refer to a subcircuit this walker "
                "has seen; run region-to-func or keep lambdas in place")
        region_attr = _attr(lambda_op, REGION_ATTR)
        if region_attr is None:
            raise LayoutError(
                "subcircuit has no {region} attribute; run "
                "assign-subcircuit-regions before the layout model")

        body = lambda_op.regions[0].blocks[0]
        # The call's wire arguments become the body's block arguments.
        args = [o for o in op.operands[1:] if _is_wire(o)]
        for arg, value in zip(body.arguments, args):
            self.vq_of[arg] = self._vq(value)

        outer, self.region = self.region, _region_index(region_attr)
        try:
            for inner in body.operations:
                self._visit(inner.operation)
        finally:
            self.region = outer

        # The body's terminator operands become the call's results.
        returned = [o for o in body.operations[len(body.operations) - 1]
                    .operation.operands if _is_wire(o)]
        for value, result in zip(returned, op.results):
            if _is_wire(result):
                self.vq_of[result] = self._vq(value)

    def _move(self, op):
        """`quake.move` -- the transfer the passes asked for, costed out."""
        vq = self._vq(op.operands[0])
        region = _region_index(_attr(op, "dest_region"))
        kind_attr = _attr(op, "dest_kind")
        # The IR names the region's array: `wires`, `in` or `out`. A move with
        # no array named is the compute wires, which is what a bare `@rN[k]`
        # meant before ports were addressable.
        named = str(kind_attr).strip('"') if kind_attr is not None else "wires"
        kind = {"wires": COMPUTE, "in": IN, "out": OUT}.get(named)
        if kind is None:
            raise LayoutError(
                f"quake.move names destination array '{named}'; expected "
                f"wires, in or out")

        if vq not in self.placement.pos:
            # First use: the qubit comes into being here rather than arriving,
            # so it is placed outright and pays no port legs.
            if kind != COMPUTE:
                raise LayoutError(
                    f"virtual qubit {vq} is first used on r{region}'s {kind}-"
                    f"port; a qubit begins life on a compute wire")
            self._place(vq, region)
        else:
            self.mover.move(vq, region, kind)
        self._thread(op)

    def _place(self, vq, region):
        """A qubit's first placement: onto a compute wire, no movement."""
        if not self.placement.has_room(region):
            raise LayoutError(
                f"the IR places virtual qubit {vq} in r{region}, which is "
                f"full (region_size={self.model.region_size})")
        # A wire another qubit has only just vacated is not free until then, so
        # the placement cannot be stamped earlier than that.
        avail = self.placement.available_at(region, COMPUTE)
        self.placement.assign(vq, region, COMPUTE)
        self.sched.ready[vq] = max(self.sched.ready_time(vq), avail)
        self.builder.delta(self.sched.time_for([vq]), "assign", vq,
                           dst=(region, COMPUTE))
        self.builder.observe_occupancy(region, self.placement.occupancy(region))

    def _gate(self, op, view):
        # Measurements carry `targets` but no `controls` group.
        controls = [self._vq(c) for c in getattr(view, "controls", [])
                    if _is_wire(c)]
        targets = [self._vq(t) for t in view.targets]
        operands = controls + targets
        params = [_constant_of(p) for p in getattr(view, "parameters", [])]

        # Placement is the passes' decision, so a gate does not move anything:
        # if its operands are not already together, the region assignment and
        # the moves disagree with the circuit and that is worth reporting, not
        # silently repairing.
        unplaced = [vq for vq in operands if vq not in self.placement.pos]
        if unplaced:
            raise LayoutError(
                f"{op.name} acts on virtual qubit(s) {unplaced} that the IR "
                f"never placed; run add-region-moves before the layout model")
        regions = {self.placement.region_of(vq) for vq in operands}
        if len(regions) > 1:
            raise LayoutError(
                f"{op.name} acts across regions {sorted(regions)}; the IR did "
                f"not move its operands together")
        off_wire = [vq for vq in operands
                    if self.placement.kind_of(vq) != COMPUTE]
        if off_wire:
            raise LayoutError(
                f"{op.name} acts on virtual qubit(s) {off_wire} parked on a "
                f"port; a gate only runs on a compute wire")

        t = self.sched.time_for(operands)
        gate = op.name.split(".", 1)[1]
        self.builder.op(t, gate, self.placement.region_of(operands[0]),
                        controls, targets, params, self.placement.pos)
        self.sched.commit(operands, t, TICK)
        if op.name in MEASUREMENTS:
            self.builder.num_measurements += 1
        self._thread(op)


def region_pipeline(model):
    """The region-lowering chain, sized from `model`.

    How many regions there are and how big they are is a property of the QPU
    being modelled, so it belongs to the model rather than to a payload -- the
    same circuit laid out on a different QPU has to be re-partitioned, not just
    re-scheduled.
    """
    # A partition may be as wide as a region can hold: any narrower and an
    # operation that would fit is split across regions for no reason.
    return ("builtin.module("
            f"func.func(outline-partitions{{strategy={PARTITIONER} "
            f"max-qubits={model.region_size}}}),"
            f"introduce-regions{{num-regions={model.num_regions} "
            f"region-size={model.region_size}}},"
            "func.func(assign-subcircuit-regions,add-region-moves)"
            ")")


def _is_region_lowered(module):
    return any(op.name == "quake.region" for op in _children(module.operation))


def lower_onto_regions(module, model):
    """Give the payload a placement, if it does not already carry one.

    The model consumes `{region = @rN}` and `quake.move`; a payload that has
    already been through the passes keeps the assignment it was lowered with,
    which is how a hand-written or pre-lowered placement can be replayed.
    """
    if _is_region_lowered(module):
        return module
    from cudaq.mlir.passmanager import PassManager
    pm = PassManager.parse(region_pipeline(model), context=module.context)
    try:
        pm.run(module.operation)
    except Exception as e:
        raise LayoutError(
            f"failed to lower the payload onto {model.num_regions} regions of "
            f"{model.region_size}: {e}") from e
    return module


def simulate_module(module, model=None):
    """Lay an already-parsed Quake module out on `model`."""
    if not module.operation.verify():
        raise LayoutError("Quake module failed verification before layout")
    model = model or QpuModel()
    entry = _find_entrypoint(module)
    if entry is None:
        raise LayoutError("no `cudaq-entrypoint` function found in payload")
    # Before lowering, not after: `outline-partitions` assumes straight-line
    # code and crashes rather than diagnosing when given a loop, so the payload
    # has to be rejected here while it still can be.
    _check_straight_line(entry)
    return Simulator(model).run(lower_onto_regions(module, model))


def simulate(mlir_text, model=None, context=None):
    """Lay a value-semantics Quake payload out on `model`. Returns a TraceBuilder."""
    ctx = context or getMLIRContext()
    return simulate_module(Module.parse(mlir_text, context=ctx), model)


def main():
    import argparse, sys
    from .trace import summarize

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("payload", nargs="?", help="Quake MLIR file (default: stdin)")
    p.add_argument("--regions", type=int, default=2)
    p.add_argument("--region-size", type=int, default=2)
    p.add_argument("-o", "--output", help="write the trace JSON here")
    p.add_argument("--viewer", metavar="PATH",
                   help="write a standalone HTML viewer with the trace embedded")
    p.add_argument("--dump-ir", metavar="PATH", nargs="?", const="-",
                   help="write the region-lowered IR the model reads, then "
                        "carry on (default: stdout)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="print only the summary line")
    args = p.parse_args()

    model = QpuModel(num_regions=args.regions,
                     region_size=args.region_size)
    src = open(args.payload).read() if args.payload else sys.stdin.read()
    if args.dump_ir:
        # Exactly what the model walks: the payload after being lowered onto
        # this model's regions, which is where every placement and move it
        # reads comes from.
        from cudaq.mlir.ir import Module as _Module
        module = lower_onto_regions(
            _Module.parse(src, context=getMLIRContext()), model)
        if args.dump_ir == "-":
            print(module)
        else:
            with open(args.dump_ir, "w") as f:
                f.write(str(module))
            print("ir: " + args.dump_ir)
    builder = simulate(src, model)
    doc = builder.to_json()
    if args.output:
        with open(args.output, "w") as f:
            f.write(builder.dumps())
    if args.viewer:
        from .viewer import write_viewer
        print("viewer: " + write_viewer(doc, args.viewer))
    if args.quiet or args.output or args.viewer:
        print(summarize(doc))
    else:
        print(builder.dumps())

