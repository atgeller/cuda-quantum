/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

struct run_test {
  __qpu__ auto operator()(const int n) {
    cudaq::qubit p;
    h(p);
    for (int i = 0; i < n; i++) {
      cudaq::qubit q;
      h(q);
      // Some operation on q depending on i
      rx(1. / (double)i, q);
      // This is not parallelizable
      x<cudaq::ctrl>(q,p);
    }
    mz(p);
  }
};

// This could be expanded to (assuming we had +4 qubits)
/*
struct run_test2 {
  __qpu__ auto operator()(const int n) {
    cudaq::qubit p;
    h(p);
    int i = 0;
    for (; i <= n - 5; i += 5) {
      cudaq::qvector q(5);
      h(q);
      // Some operation on q depending on i
      rx(1. / (double)i, q[0]);
      rx(1. / (double)i+1, q[1]);
      rx(1. / (double)i+2, q[2]);
      rx(1. / (double)i+3, q[3]);
      rx(1. / (double)i+4, q[4]);
      x<cudaq::ctrl>(q[0],p);
      x<cudaq::ctrl>(q[1],p);
      x<cudaq::ctrl>(q[2],p);
      x<cudaq::ctrl>(q[3],p);
      x<cudaq::ctrl>(q[4],p);
    }
    for (; i < n; i++) {
      cudaq::qubit q;
      h(q);
      // Some operation on q depending on i
      rx(1. / (double)i, q);
      x<cudaq::ctrl>(q,p);
    }
    mz(p);
  }
};*/

int main() {
  auto counts = cudaq::sample(run_test{}, 20);
  counts.dump();
  return 0;
}