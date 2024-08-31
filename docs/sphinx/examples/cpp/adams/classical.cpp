/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <iostream>

struct run_test1 {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    double v[2] = {};
    v[0] = 1.;

    rx(v[0],q);
  }
};
/*
struct run_test2 {
  __qpu__ auto operator()() {
    cudaq::qubit q,p,r;
    double i = 0.;

    h(q);
    h(r);
    double j = i + 1.;
    h(p);
    x<cudaq::ctrl>(q,p);
    // Reset q
    h(q);
    // Ops on p
    double k = i * 1.;
    z(p);
    ry(k,p);
    x<cudaq::ctrl>(r,p);
    // Reset r
    h(r);
    // Measure p
    mz(p);
  }
};*/

int main() {
  auto estimate = cudaq::sample(run_test1{});
  //counts.dump();
  return 0;
}
