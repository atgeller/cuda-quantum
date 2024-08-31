/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <iostream>

struct run_test {
  __qpu__ void operator()() {
    cudaq::qubit q;
    cudaq::qubit p;
    h(q);
    while(!mz(q)) {
        x(p);
        rz(1,q);
    }
    mz(q);
    mz(p);
  }
};

// Without else branch
// q: -- h(0, 0) -- join(1) -- mz(2) -- rz(2, 0) -- mz(3)
// p: --            join(1) --       -- x(2, 1)  -- mz(4)

int main() {
  auto counts = cudaq::sample(run_test{});
  counts.dump();
  return 0;
}