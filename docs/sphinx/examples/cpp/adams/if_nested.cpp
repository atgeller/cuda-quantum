/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

struct run_test {
  __qpu__ auto operator()(const int n) {
    cudaq::qubit q;
    cudaq::qubit p;
    cudaq::qubit r;

    h(q);
    if (mz(q)) {
      h(p);
      if (mz(p))
        h(p);
    } else {
      h(p);
      y(p);
    }
    
    mz(p);
  }
};

// Without else branch
// q: -- h(0, -1) -- mz(0) --         -- cx(1) -- join(2) -- mz(3)
// q:                      --                  -- join(2) -- mz(3)
// p: --                   -- h(1,-1) -- cx(1) -- join(2)
// p:                                          -- join(2)

// With else branch
// q: -- h(0, -1) -- mz(0) --         -- cx(1) -- join(2) -- mz(3)
// q:                      --                  -- join(2) -- mz(3)
// p: --                   -- h(1,-1) -- cx(1) -- join(2)
// p:                      -- h(1,-1) -- mz(1) -- join(2)

int main() {
  auto counts = cudaq::sample(run_test{}, true);
  counts.dump();
  return 0;
}
