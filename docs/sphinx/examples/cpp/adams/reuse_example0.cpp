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
    cudaq::qubit r;

    // Ops on q
    h(q);
    // Reset q
    //if (mz(q))
    mz(q);
      x(q);
    // Ops on p
    h(p);
    x(p);
    y(p);
    z(p);
    // Ops on r
    h(r);
    x<cudaq::ctrl>(r,p);
    // Reset p
    //if (mz(p))
    mz(p);
      x(p);
    // Reset r
    //if (mz(r))
    mz(r);
      x(r);
  }
};

int main() {
  auto counts = cudaq::sample(run_test{});
  return 0;
}