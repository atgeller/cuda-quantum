#!/bin/bash

# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

docker build -t cudaq-dev -f docker/build/cudaq.dev.Dockerfile . --build-arg install="CMAKE_BUILD_TYPE=Release CUDA_QUANTUM_VERSION=latest"

nvidia-docker run -dit --name runner --rm cudaq-dev 

docker exec runner python3 -u benchmarks/run_phase_folding_benchmarks.py --block-lengths 5 10 25 50 100 250 500 1000 --rz-weights 0.15 0.3 0.5 0.7 0.85 --n-qubits 3 5 10 15 20 --seed 489 --n-seeds=5 --iterations=3 --raw-data-file=raw.csv

docker cp runner:/workspaces/cuda-quantum/results.csv .
docker cp runner:/workspaces/cuda-quantum/raw.csv .

docker stop runner
