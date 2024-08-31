/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

struct run_test {
  __qpu__ void operator()(const int n_qubits) {
    for (int i = 0; i < n_qubits; i++) {
      cudaq::qvector q(n_qubits);
      h(q);
      if (!mz(q[i])) {
        break;
      }
    }
  }
};

int main() {
  auto counts = cudaq::sample(run_test{}, 5);
  counts.dump();
  return 0;
}
