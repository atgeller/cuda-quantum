/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

template <std::size_t N>
struct ghz {
  double operator()() __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    auto res = mz(q[1]);
    double returnVal = 0.0;
    if (res)
      returnVal = 1.0;
    return returnVal;
  }
};

int main() {

  double result = ghz<10>{}();
  printf("Result = %f\n", result);

  return 0;
}