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
    cudaq::qubit q,p;

    h(q);
    bool condition = mz(q);
    if (condition) {
      y(p);
      x<cudaq::ctrl>(p,q);
    } else {
      z(p);
      x<cudaq::ctrl>(p,q);
    }
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(run_test{});
  counts.dump();
  return 0;
}
