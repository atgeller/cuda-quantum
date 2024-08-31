/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

struct run_test {
  __qpu__ void operator()(int n) {
    cudaq::qubit q;
    cudaq::qubit p;
    h(q);
    for (int i = 0; i < n; i++) {
      rx(1.,p);
    }
    x<cudaq::ctrl>(p,q);
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(run_test{}, 5);
  counts.dump();
  return 0;
}
