/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

__qpu__ void f() {
    cudaq::qubit q;

    h(q);
    if (mz(q)) {
        f();
    }
}

struct run_test {
  __qpu__ auto operator()() {
    f();
  }
};

__qpu__ void f2();

__qpu__ void f1() {
    cudaq::qubit q;

    h(q);
    if (mz(q)) {
        f2();
    }
}

__qpu__ void f2() {
    cudaq::qubit q;

    h(q);
    if (mz(q)) {
        f1();
    }
}

struct run_test2 {
  __qpu__ auto operator()() {
    f1();
  }
};

int main() {
  // Neither run_test nor run_test2 are accepted here
  auto counts = cudaq::sample(run_test2{});
  counts.dump();
  return 0;
}