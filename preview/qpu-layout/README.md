# QPU layout preview

A CUDA-Q target that lays a kernel out on a region-based QPU and records what it
costs. It does **not** simulate quantum state -- counts come back all-zero. The
trace is the result.

> **This preview is entirely agentic-driven.** Every line of it -- the model, the
> viewer, the tests and this README -- was written by an AI agent working from a
> human's design decisions. It has not had a line-by-line human review. Treat it
> as a research artifact for exploring region-based layout, not as production
> code, and read it before trusting a number it produces.

## What decides what

Placement and movement belong to the compiler passes. The layout model only
decides *when* things happen and writes the trace:

| Decision | Made by |
| --- | --- |
| Which subcircuit runs in which region | `assign-subcircuit-regions` |
| When a qubit leaves, crosses, arrives | `add-region-moves` |
| Which wire inside a region | nobody -- a region is all-to-all |
| Which timestep each operation runs in | the model (ASAP schedule) |

The model rejects IR it cannot account for rather than repairing it: a gate on a
qubit no region placed, operands the passes never brought together, or a region
assignment too large for the QPU it is being laid out on.

Every operation takes one tick -- a gate, and each leg of a move alike. There is
no relative cost model yet.

## Using it

```bash
export PYTHONPATH="$PYTHONPATH:/workspaces/cuda-quantum/preview/qpu-layout/python"
```

Through a target:

```python
import cudaq
from cudaq_qpu_layout import QpuLayoutTarget

target = QpuLayoutTarget.build(num_regions=2, region_size=4)
cudaq.set_target(target)
cudaq.sample(my_kernel)

trace = target.runtime_endpoint.trace
print(trace["summary"])
```

Or on a Quake payload directly, with no target and no compilation:

```bash
python3 -m cudaq_qpu_layout payload.mlir --regions 2 --region-size 4 \
    -o trace.json --viewer trace.html
```

The payload does not need to carry a region assignment: the model lowers it onto
its own regions first. How many regions there are, and how big, is a property of
the QPU being modelled, so the same circuit can be laid out on several.

`--dump-ir PATH` writes the region-lowered IR the model actually walks, which is
where every placement and move in the trace comes from.

## Starting the visualizer

```bash
export PYTHONPATH="$PYTHONPATH:/workspaces/cuda-quantum/preview/qpu-layout/python"
python3 -m cudaq_qpu_layout.serve --port 8765 --dir <trace dir>
```

Then, on the host, `http://localhost:8765/`. Port 8765 is forwarded by
`.devcontainer/devcontainer.json`.

Three ways in:

- **`/`** -- paste a CUDA-Q kernel, choose the QPU shape, and get its layout.
  The kernel must take no arguments and stand alone: CUDA-Q lifts module-level
  values and calls to sibling kernels into runtime arguments, which leaves
  nothing concrete to lay out. Loops are unrolled for you.
- **`/viewer.html?trace=NAME.json`** -- a trace written into the served
  directory.
- **`/NAME.html`** -- a standalone viewer with a trace baked in, from
  `--viewer` or `write_viewer(trace, path)`. These open without a server at all.

In the viewer, regions are a grid inside a zoomable viewport; the controls stay
a constant size as you zoom out to take in a wide QPU. `v` cycles qubit
residence, gate labels and the circuit timeline; arrow keys step; `+`/`-`/`0`
zoom.

Submitting a kernel **executes the submitted Python on the server**. That is
inherent to accepting CUDA-Q source, but `serve.py` binds `0.0.0.0` so it is
reachable by anything that can route to the container. Fine inside a dev
container; do not expose it further.

## Partitioners

`outline-partitions` decides which operations become a subcircuit, which is what
the region assignment then places.

- **`greedy`** (default) weighs every operation.

They cost the same on the circuits tried so far. Select one with
`--outline-partitions='strategy=greedy-pair'`, or `QPU_LAYOUT_PARTITIONER` for
the layout model's own lowering.

## Layout

| Path | Role |
| --- | --- |
| `model.py` | `QpuModel`: how many regions, and how many wires each |
| `sim.py` | the walker, the schedule, and the lowering onto a model's regions |
| `trace.py` | trace schema and `replay()` |
| `target.py` | the compile target, runtime endpoint and `CustomTarget` |
| `fromsource.py` | CUDA-Q source in, trace out; what the submit page runs |
| `viewer.py`, `viewer.html`, `submit.html`, `serve.py` | the web front end |

There is no `.yml`, no C++ `ServerHelper`, no mock HTTP server, and nothing to
build -- this preview is pure Python on top of an installed CUDA-Q. The passes it
drives are ordinary `cudaq-opt` passes and do need a build.

## Not modeled yet

Intra-region topology: a region is all-to-all, so co-located qubits interact at
no cost, and no wire inside a region is ever named. When position is modeled it
belongs in the passes as a **move** -- which may or may not lower to a SWAP --
never as a swap primitive.

Also absent: Clifford-frame merging across joined regions, heterogeneous region
sizes, magic-state factories, and classically-conditioned operations. Ports are
unbounded: they cost but never queue, so only compute wires are scarce.

`summary.moves.cross` counts a subcircuit re-entering the region it just left,
so it is not purely a count of inter-region transfers.
