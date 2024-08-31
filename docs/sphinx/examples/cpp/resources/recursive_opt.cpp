/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

__qpu__ auto f(cudaq::qubit &p, int i) {
    cudaq::qubit q;
    h(q);
    // Some operation on q depending on i
    rx(1. / (double)i, q);
    x<cudaq::ctrl>(q,p);

    if (i == 0) return 0;
    return f(p, i - 1) + 1;
}

__qpu__ auto f_opt(cudaq::qubit &p, int i) {
    cudaq::qvector q(5);
    h(q);
    // Some operation on q depending on i
    rx(1. / (double)i, q);
    x<cudaq::ctrl>(q[0],p);
    x<cudaq::ctrl>(q[1],p);
    x<cudaq::ctrl>(q[2],p);
    x<cudaq::ctrl>(q[3],p);
    x<cudaq::ctrl>(q[4],p);
    // Could collapse to x<cudaq::ctrl>(q,p);

    if (i < 5) return f(p,i);
    return f_opt(p, i -= 5) + 5;
}

struct run_test {
  __qpu__ auto operator()(const int n) {
    cudaq::qubit p;
    h(p);
    f(p, 5);
    mz(p);
  }
};