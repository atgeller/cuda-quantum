/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// This program should (not) be illegal, but currently the error is a result of a violated assumption in LowerToCFG when doing a dynamic cast
struct run_test {
  __qpu__ auto operator()(const int n_qubits) {
    for (int i = 1; i < n_qubits; i++) {
      cudaq::qvector q(i);
      h(q);
      // Avoid break statement for now
      if (!mz(q[i-1])) {
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

// struct run_test {
//   __qpu__ auto operator()() {
//     // Move qubit allocation out here and no problem
//     for (int i = 0; i < 5; i++) {
//       cudaq::qubit q;
//       if (mz(q)) {
//         break;
//       }
//     }
//   }
// };

// int main() {
//   auto counts = cudaq::sample(run_test{});
//   counts.dump();
//   return 0;
// }
