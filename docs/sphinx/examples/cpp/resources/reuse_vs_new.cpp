/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

struct run_test {
  __qpu__ auto operator()(const int n) {
    cudaq::qubit q;
    cudaq::qubit p;

    // Do some computation over q
    h(q);

    // Happen to measure q anyway
    auto res = mz(q);

    // Still use p
    h(p);
    mz(p)
  }
};

struct run_test_opt {
  __qpu__ auto operator()(const int n) {
    cudaq::qubit q;

    h(q);

    auto res = mz(q);

    // Reset and reuse q instead
    if (res) x(q);

    h(q);
    mz(q)
  }
};

int main() {
  auto counts = cudaq::sample(run_test{}, 20);
  counts.dump();
  return 0;
}
