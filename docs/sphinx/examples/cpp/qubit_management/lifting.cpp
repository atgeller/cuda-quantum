/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

__qpu__ auto test1() {
  cudaq::qubit q,p;

  // Cycle 0
  x<cudaq::ctrl>(q,p);
  // Cycle 1
  auto res = mz(q);
  if (res) {
    // Cycle 2
    y(p);
    // Cycle 3
    x(p);
  } else {
    // Cycle 2
    y(p);
    // Cycle 3
    z(p);
  }

  mz(p);
}

__qpu__ auto test2() {
  cudaq::qubit q,p;

  // Cycle 0
  h(q);

  // Cycle 1
  h(p);
  auto res = mz(q);

  if (res) {
    // Cycle 2
    y(p);
    // Cycle 3
    x(p);
  } else {
    // Cycle 2
    y(p);
    // Cycle 3
    z(p);
  }

  mz(p);
}

__qpu__ auto test3() {
  cudaq::qubit q,p,r;

  // Cycle 0
  x<cudaq::ctrl>(r,q);
  h(p);

  // Cycle 1
  auto res = mz(q);
  y(p);

  if (res) {
    cudaq::qubit s;
    y(s);
    // Cycle 0 -3-
    x(p);
  } else {
    cudaq::qubit t;
    y(t);
    // Cycle 0 -3-
    z(p);
  }

  mz(p);
}