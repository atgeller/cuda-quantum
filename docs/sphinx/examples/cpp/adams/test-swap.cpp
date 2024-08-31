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
    //cudaq::qvector qubits(2);
    cudaq::qubit p,q;

    x(q);
    x(p);
    x<cudaq::ctrl>(q,p);
    x(q);
    x(q);

    // Dependency analysis changes the order of measurements which changes the output
    // Even though the results register name stays the same
    auto result1 = mz(q);
    auto result2 = mz(p);
  }
};

int main() {
  auto result = cudaq::sample(run_test{});
  std::cout << result.most_probable() << '\n';
  return 0;
}
