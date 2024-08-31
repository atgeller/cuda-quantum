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

    h(q);
    // Measurement is slow
    if (mz(q)) {
        // Some computation on p
        x(p);
        h(p);
    } else {
        // Other computation on p
        h(p);
    }

    // Use p
    mz(p);
  }
};

struct run_test_opt {
  __qpu__ auto operator()() {
    cudaq::qubit q;
    cudaq::qubit p;
    cudaq::qubit r;
    cudaq::qubit* res;

    h(q);
    x(p);
    h(p);
    h(r);
    // Think about two branches, one parallelizable, one not
    if (mz(q)) {
        res = &p;
    } else {
        res = &r;
    }

    // Use result
    mz(*res);
  }
};

int main() {
  auto counts = cudaq::sample(run_test_opt{});
  counts.dump();
  return 0;
}
