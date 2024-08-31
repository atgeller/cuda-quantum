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

    // Use measurement of q anyway
    auto res = mz(q);
    if (res) {
      h(p);
    } //...

    // Do something with p, q no longer in use
    x(p);

    // ...
  }
};
/*
struct run_test_opt {
  __qpu__ auto operator()() {
    cudaq::qubit q;
    cudaq::qubit p;
    h(q);

    // Use measurement of q anyway
    auto res = mz(q);
    // Now, just reset q
    if (res)
      x(q);
      //...
    
    x(q);

    // ...
  }
};*/

int main() {
  auto counts = cudaq::sample(run_test{});
  counts.dump();
  return 0;
}
