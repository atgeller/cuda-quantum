/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// This doesn't work, can't copy construct to return
// Caught by C++
// __qpu__ cudaq::qubit even(int num) {
//   if (num == 0) {
//     return cudaq::qubit();
//   }
//   return odd(num - 1).negate();
// }
// __qpu__ cudaq::qubit odd(int num) {
//   if (num == 0) {
//     return cudaq::qubit().negate();
//   }
//
//   return even(num - 1).negate();
// }

// Both the following coin flip tests work
// struct run_test {
  // template <typename CallableKernel>
  // __qpu__ auto operator()(const int n_qubits, const int n_iterations,
  //                         CallableKernel &&oracle) {

  //   int sum = 0;
  //   // Legal: q goes out of scope and so safe to reuse
  //   for (int i = 0; i < n_iterations; i++) {
  //     for (int i = 0; i < n_qubits; i++) {
  //       cudaq::qubit q;

  //       h(q);
  //       if (mz(q)) {
  //         sum++;
  //       }
  //     }
  //   }
  // }
/*
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, const int n_iterations,
                          CallableKernel &&oracle) {

    int sum = 0;
    
    // Coin toss n times
    cudaq::qvector q(n_qubits);
    for (int i = 0; i < n_iterations; i++) {
      for (int i = 0; i < n_qubits; i++) {
        h(q[i]);
        if (mz(q[i])) {
          sum++;
        }
      }
    }

    // Coin toss n times again
    cudaq::qvector p(n_qubits);
    for (int i = 0; i < n_iterations; i++) {
      for (int i = 0; i < n_qubits; i++) {
        h(q[i]);
        if (mz(q[i])) {
          sum++;
        }
      }
    }
  }
};*/

// This works
// struct run_test {
//   __qpu__ auto operator()(const int n_qubits) {
//     cudaq::qvector q(n_qubits);
//     for (int i = 0; i < n_qubits; i++) {
//       h(q[i]);
//       if (!mz(q[i])) {
//         break;
//       }
//       mz(q);
//     }
//   }
// };

// Works because it's a qview, not a qvector
// __qpu__ int f(int position, cudaq::qview<> q) {
//   int res = 0;
//   if (q.size() > 1) {
//     auto splitpoint = (q.size() + 1) / 2;
//     // Two non-overlapping halves
//     auto q1 = q.slice(0, splitpoint);
//     auto q2 = q.slice(splitpoint, q.size() - splitpoint);

//     // recursive call
//     res += f(position*2, q1);
//     res += f(position*2+1, q2);
//   } else {
//     // Some computation
//     h(q.front());
//     if (mz(q.front())) {
//       res = position;
//     }
//   }

//   return res;
// }

// int main() {
//   cudaq::qvector q(10);
//   auto sum = f(0, q);
//   printf("Sum: %d\n", sum);
// }

// Doesn't like that p is being returned (presumably because it's leaving scope)
// __qpu__ cudaq::qubit my_op(cudaq::qubit q) {
//   cudaq::qubit p;
//   x(q,p);
//   return p;
// }

// Can't pass by value either
// __qpu__ bool my_op(cudaq::qubit q) {
//   cudaq::qubit p;
//   h(p);
//   x(q,p);
//   return mz(p);
// }

// int main() {
//   cudaq::qubit q;
//   h(q);
//   my_op(q);
// }

// Pass by reference is fine though
// __qpu__ bool my_op(cudaq::qubit& q) {
//   cudaq::qubit p;
//   x(q,p);
//   return mz(p);
// }

// int main() {
//   cudaq::qubit q;
//   h(q);
//   my_op(q);
// }

// __qpu__ bool my_op() {
//   cudaq::qubit p;
//   int res = 0;
//   h(p);
//   if (mz(p)) {
//     cudaq::qubit q;
//     x(p,q);
//     res = mz(q);
//   } else {
//     cudaq::qubit y,z;
//     h(y);
//     x(y,z);
//     res = mz(z);
//   }
//   return mz(p);
// }

__qpu__ void test2() {
  cudaq::qubit p;
  h(p);
  mz(p);
}

// This works
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;
    h(q);
    if (mz(q)) {
      test2();
    }
  }
};

int main() {
  auto counts = cudaq::sample(run_test{});
  counts.dump();
  return 0;
}
