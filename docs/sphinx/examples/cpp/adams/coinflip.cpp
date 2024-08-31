/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>
#include <iostream>

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    h(q);
    if (mz(q)) {
      cudaq::qubit p;
      h(p);
      mz(p);
    }
  }
};

int main() {
  auto estimate = cudaq::sample(run_test{});
  //counts.dump();
  return 0;
}
