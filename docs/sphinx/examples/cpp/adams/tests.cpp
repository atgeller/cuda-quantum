/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

struct run_test1 {
  __qpu__ auto operator()() {
    cudaq::qubit q;
    h(q);
    mz(q);
  }
};

struct run_test2 {
  __qpu__ auto operator()() {
    cudaq::qubit q;
    cudaq::qubit p;
    h(p);
    h(q);
    mz(q);
  }
};

struct run_test3 {
  __qpu__ auto operator()() {
    cudaq::qubit q;
    cudaq::qubit p;
    h(p);
    x<cudaq::ctrl>(p, q);
    y(q);
    mz(q);
  }
};

struct run_test4 {
  __qpu__ auto operator()() {
    cudaq::qubit q1;
    cudaq::qubit q2;
    cudaq::qubit q3;
    cudaq::qubit q4;
    cudaq::qubit q5;
    h(q1);
    h(q2);
    x<cudaq::ctrl>(q1, q2);
    h(q3);
    x<cudaq::ctrl>(q2, q3);
    h(q4);
    x<cudaq::ctrl>(q3, q4);
    h(q5);
    x<cudaq::ctrl>(q4, q5);
    mz(q5);
  }
};

struct run_test5 {
  __qpu__ auto operator()() {
    cudaq::qubit q1;
    cudaq::qubit q2;
    cudaq::qubit q3;
    cudaq::qubit q4;
    cudaq::qubit q5;
    h(q1);
    h(q2);
    x<cudaq::ctrl>(q1, q2);
    h(q3);
    x<cudaq::ctrl>(q2, q3);
    h(q4);
    x<cudaq::ctrl>(q3, q4);
    h(q5);
    h(q5);
    h(q5);
    h(q5);
    h(q5);
    h(q5);
    x<cudaq::ctrl>(q4, q5);
    mz(q5);
  }
};

struct run_test6 {
  __qpu__ auto operator()() {
    cudaq::qubit q,p,r;

    h(r);
    x(r);
    y(r);
    h(p);
    x<cudaq::ctrl>(q,p);
    x<cudaq::ctrl>(q,r);
    z(p);
    y(p);
    x(p);
    mz(p);
  }
};

struct run_test7 {
  __qpu__ auto operator()() {
    cudaq::qubit q;
  }
};

int main() {
  //auto estimate = cudaq::sample(run_test1{});
  //counts.dump();
  return 0;
}