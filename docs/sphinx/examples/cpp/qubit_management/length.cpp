/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// nvq++ --target opt-test --target-option dep-analysis,qpp examples/cpp/qubit_management/length.cpp

struct run_test {
  __qpu__ bool operator()() {
    cudaq::qubit q;

    h(q);
    bool condition = true;
    if (condition) {
      cudaq::qubit p;
      h(p);
      x<cudaq::ctrl>(p,q);
    } else {
      cudaq::qubit r;
      h(r);
      y<cudaq::ctrl>(r,q);
    }
    return mz(q);
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
