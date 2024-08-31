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
    bool condition = true;

    cudaq::qubit p;

    bool b = false;

    if (condition) {
      cudaq::qubit q,r;
      // Cycle 1
      x<cudaq::ctrl>(p,q);

      // Cycle 2
      h(p);
      h(r);

      // Cycle 3
      y(p);
      b = mz(r);
    } else {
      y(p);   
    }
    return b && mz(p);
  }
};

int main() {
  return 0;
}
