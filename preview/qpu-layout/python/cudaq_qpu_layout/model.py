# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The QPU model the layout simulator places a circuit onto.

A QPU is a set of uniform `regions`, each holding `region_size` logical qubits
in numbered compute slots. Entangling two qubits requires them to be in the same
region; getting them there costs moves, which is what the simulator measures.
Intra-region topology is not modeled -- a region is all-to-all, so co-located
qubits interact directly.

A region is entered and left only through its ports: a qubit leaves a compute
wire onto an out-port, crosses to the destination region's in-port, and lands on
a compute wire there -- the three-leg calling convention of
`LoweringQubitsToRegions.md`. Ports are modeled as unbounded, so they contribute
cost but never serialize movement; only compute slots are a scarce resource.

Every operation takes one tick -- a gate, and each leg of a move alike. There is
no relative cost model yet, so depth counts operations rather than weighting
them; a crossing being dearer than a gate is a calibration to make once the
rest of the model is settled.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QpuModel:
    num_regions: int = 2
    region_size: int = 2

    def __post_init__(self):
        if self.num_regions < 1:
            raise ValueError("num_regions must be at least 1")
        if self.region_size < 1:
            raise ValueError("region_size must be at least 1")

    @property
    def capacity(self):
        return self.num_regions * self.region_size

    def to_json(self):
        return {
            "num_regions": self.num_regions,
            "region_size": self.region_size,
        }

    @staticmethod
    def from_json(d):
        """Build a model from a dict, coercing the string values a target
        configuration delivers (`cudaq.set_target(..., num_regions=4)`)."""
        fields = QpuModel.__dataclass_fields__
        kwargs = {k: int(v) for k, v in d.items() if k in fields}
        return QpuModel(**kwargs)
