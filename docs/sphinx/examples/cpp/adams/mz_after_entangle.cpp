/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;
    cudaq::qubit p;

    // 0
    h(q);
    // 0
    h(p);
    // 1
    x<cuda::ctrl>(q,p);
    // 2
    y(p);
    // 3
    z(p);
    // At 2 or at 4?
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(run_test{});
  return 0;
}