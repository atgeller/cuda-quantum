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
    cudaq::qubit q,p;
    float f;
    
    /*
    h(q);
    y(q);
    if (mz(q))
      f = 1.;
    else
      f = 2.;

    if (true) {
      rx(f, p);
    } else {
      ry(f, p);
    }*/

    bool b = mz(q);
    f = (float)b * 5.;

    if (true) {
      rx((float)b * 2., p);
    } else {
      ry((float)b + 1., p);
    }

    return mz(p);
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
