# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Lay a CUDA-Q Python kernel out, starting from its source text.

Reads a module on stdin, finds the kernel to lay out, and writes the trace as
JSON on stdout. Run as its own process so that a kernel which fails to compile,
or brings down the CUDA-Q runtime, cannot take the server with it.

    echo "$code" | python3 -m cudaq_qpu_layout.fromsource --regions 4

`@cudaq.kernel` reads a kernel's source with `inspect`, so the text has to
reach a real file on disk; a string handed to `exec` has no source to find.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile

from .model import QpuModel
from .sim import simulate_module, LayoutError
from .target import UNROLL_PIPELINE, WIRESET_PIPELINE


class SourceError(RuntimeError):
    """The submitted source did not yield a kernel that can be laid out."""


def _kernels(module):
    """The CUDA-Q kernels a module defines, in definition order."""
    found = []
    for name in dir(module):
        obj = getattr(module, name)
        if type(obj).__name__ == "PyKernelDecorator":
            found.append((name, obj))
    return found


def _choose(kernels, entry):
    if not kernels:
        raise SourceError(
            "no @cudaq.kernel found. Decorate the kernel to lay out with "
            "@cudaq.kernel.")
    if entry:
        for name, kernel in kernels:
            if name == entry:
                return name, kernel
        raise SourceError(
            f"no kernel named '{entry}'; this module defines " +
            ", ".join(n for n, _ in kernels))
    # A kernel that takes arguments cannot be laid out on its own, so prefer
    # one that takes none -- that is the entry point of a self-contained module.
    def arity(kernel):
        try:
            return kernel.formal_arity()
        except Exception:
            return 0

    nullary = [(n, k) for n, k in kernels if arity(k) == 0]
    if not nullary:
        raise SourceError(
            "every kernel here takes arguments; the one to lay out must take "
            "none. Add a wrapper kernel that calls it with concrete values.")
    return nullary[-1]


def quake_of(source, entry=None):
    """Compile submitted source and return its kernel's Quake."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "submitted_kernel.py")
        with open(path, "w") as f:
            f.write(source)
        spec = importlib.util.spec_from_file_location("submitted_kernel", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["submitted_kernel"] = module
        try:
            # Whatever the module prints goes to stderr: stdout carries the
            # trace back to the caller and must stay machine-readable.
            with contextlib.redirect_stdout(sys.stderr):
                spec.loader.exec_module(module)
        except Exception as e:
            raise SourceError(f"{type(e).__name__}: {e}") from e

        name, kernel = _choose(_kernels(module), entry)
        mlir = str(kernel)
        if "quake.pylifted" in mlir:
            raise SourceError(
                f"kernel '{name}' captures module-level values or other "
                "kernels, which CUDA-Q lifts into runtime arguments. Inline "
                "them, or use literals, so the kernel stands alone.")
        return name, mlir


def trace_of(source, model, entry=None):
    """Lay the submitted module's kernel out on `model`."""
    from cudaq.mlir.ir import Module
    from cudaq.mlir.passmanager import PassManager
    from cudaq.kernel.utils import getMLIRContext

    name, mlir = quake_of(source, entry)
    module = Module.parse(mlir, context=getMLIRContext())
    # Straight from Python: reference semantics, and the loops still rolled
    # up. `prepare-for-wireset` gives each qubit its own wire -- the partitioner
    # backs out of anything aliasable -- and the unroll leaves straight-line
    # code, which is all the layout model schedules.
    for stage, pipeline in (("prepare", "builtin.module(" +
                             WIRESET_PIPELINE + ")"),
                            ("unroll", UNROLL_PIPELINE)):
        try:
            PassManager.parse(pipeline,
                              context=module.context).run(module.operation)
        except Exception as e:
            raise SourceError(f"could not {stage} '{name}': {e}") from e
    return name, simulate_module(module, model).to_json()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regions", type=int, default=2)
    p.add_argument("--region-size", type=int, default=4)
    p.add_argument("--entry", help="which kernel to lay out")
    args = p.parse_args()

    source = sys.stdin.read()
    try:
        name, trace = trace_of(
            source,
            QpuModel(num_regions=args.regions, region_size=args.region_size),
            args.entry)
    except (SourceError, LayoutError) as e:
        json.dump({"error": str(e)}, sys.stdout)
        return 1
    except Exception as e:  # noqa: BLE001 - the browser needs to see anything
        json.dump({"error": f"{type(e).__name__}: {e}"}, sys.stdout)
        return 1
    json.dump({"kernel": name, "trace": trace}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
