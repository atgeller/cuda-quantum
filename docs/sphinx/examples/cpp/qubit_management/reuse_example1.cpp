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
    cudaq::qubit q,p,r;

    h(q);
    h(p);
    h(r);
    x<cudaq::ctrl>(q,p);
    // Ops on p
    y(p);
    z(p);
    x<cudaq::ctrl>(r,p);
    // Reset q
    h(q);
    // Reset r
    h(r);
    // Measure p
    mz(p);
  }
};

int main() {
  auto counts = cudaq::sample(run_test{});
  counts.dump();
  return 0;
}