/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// Move after if
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p;

    // Want to move cnot after if.
    h(q);
    bool condition = true;
    if (condition) {
      y(p);
      x<cudaq::ctrl>(p,q);
    } else {
      z(p);
      x<cudaq::ctrl>(p,q);
    }
    mz(q);
  }
};

// Parallelization by splitting
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p;

    // Can split p between if/then branches to run with the mz/h on q
    h(q);
    bool condition = mz(q);
    if (condition) {
      y(p);
      x<cudaq::ctrl>(p,q);
    } else {
      z(p);
      x<cudaq::ctrl>(p,q);
    }
    mz(q);
  }
};

// Parallelization by splitting
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p;

    // Can split p between if/then branches to run with the mz/h on q
    h(q);
    bool condition = mz(q);
    if (condition) {
      y(p);
      y(q);
      x(p);
      //x(q);
    } else {
      z(p);
      z(q);
      h(p);
      //x(q);
    }

    x(q);
    mz(q);
  }
};

// Reuse allows moving out of if
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,s;

    h(q);
    h(s);
    bool condition = mz(q);
    // Choosing same fresh alloc for p/r would allow moving the h(p/r)
    // before the if to run with the mz(q)/h(q)
    if (condition) {
      cudaq::qubit p;
      h(p);
      y(p);
      x(s);
      x<cudaq::ctrl>(p,s);
    } else {
      cudaq::qubit r;
      h(r);
      y(r);
      z(s);
      x<cudaq::ctrl>(r,s);
    }
    mz(s);
  }
};

// Moving before if but after measure, removes qubit from if
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p;

    x(p);
    y(p);
    z(p);
    h(q);
    bool condition = mz(q);
    // We can move x(q) before the if as long as it's after the measure
    if (condition) {
      x(q);
      z(p);
    } else {
      x(q);
      y(p);
    }
    mz(p);
  }
};

// Unbalanced nested ifs, can reuse
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p,r,s;

    h(q);
    h(s);
    // Want to parallelize z, use same physical qubit for p/r
    if (mz(q)) {
      if (mz(s)) {
        z(r);
        mz(r);
      }
    } else {
      z(p);
      mz(p);
    }
  }
};

// De-parallelization example
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p,r;

    h(p);
    h(r);
    h(q);
    z(p);
    z(r);

    // Seems simple that p/r can be mapped to same, but hard to tell with
    // current setup
    if (mz(q)) {
      x(r);
      mz(r);
    } else {
      y(p);
      mz(p);
    }
  }
};

// Unbalanced nested ifs with different ops
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    h(q);

    if (mz(q)) {
      cudaq::qubit p;
      h(p);

      if (mz(p)) {
        cudaq::qubit r,s;
        h(r);
        h(s);
        x<cudaq::ctrl>(r,s);
        mz(s);
      } else {
        cudaq::qubit r;
        y(r);
        mz(r);
      }
    } else {
      cudaq::qubit r,s;
      h(r);
      h(s);
      mz(r);
      mz(s);
    }

    y(q);
  }
};

// Unbalanced nested ifs with similar ops
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    h(q);

    if (mz(q)) {
      h(q);
      if (mz(q)) {
        cudaq::qubit p;
        h(p);
        x(p);
        mz(p);
      }
    } else {
      cudaq::qubit p;
      h(p);
      x(p);
      mz(p);
    }

    y(q);
  }
};

// Balanced nested ifs with similar ops
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    h(q);

    if (mz(q)) {
      h(q)
      if(mz(q)) {
        cudaq::qubit p;
        h(p);
        mz(p);
      } else {
        cudaq::qubit p;
        x(p);
        mz(p);
      }
    } else {
      h(q)
      if(mz(q)) {
        cudaq::qubit p;
        x(p);
        mz(p);
      } else {
        cudaq::qubit p;
        h(p);
        mz(p);
      }
    }

    y(q);
  }
};

// Can use extra qubit to optimize one branch
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p;

    h(q);
    h(p);

    if (mz(q)) {
      cudaq::qubit r;
      y(p);
      h(r);
      x<cudaq::ctrl>(r,p);
      mz(p);
    } else {
      cudaq::qubit r;
      z(r);
      y(r);
      x(r);
      mz(r);
    }
  }
};

// Can use extra qubit to optimize one branch
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q,p;

    h(q);
    h(p);

    if (mz(q)) {
      cudaq::qubit r;
      y(p);
      h(r);
      x<cudaq::ctrl>(r,p);
      mz(p);
    } else {
      cudaq::qubit r;
      z(r);
      y(r);
      x(r);
      mz(r);
    }
  }
};

/*
 * Strategy:
    first move allocations to inner scopes whenever possible
    Mkae a decision on qubit reuse (for allocation within that branch) within the each child/inner scope of a branch separately 
      -> Approximation 1:  creating a square space-time for each block could be suboptimal compared to "trying to fit puzzle pieces", but its an acceptable goal to make the optimization processable
 * for any qubirts not allocated within the child branch, move out any instructions on these qubits that match across all branches (same for beginning of branch -> move up, and end of branch -> move down)
 * Observation 1: if we start computing a branch before we know the condition by using extra qubits, we have no good options to clean up the extra qubits if that branch is then not taken. Hence, using extra qubits for the sake of starting to compute an if is in general (except in trivial cases) not a good option. 
 * Ergo from Observation 1: The only case where we can start to execute the branch early (for instructions that haven't already been moved out of the branch) is if the instruction across all branches for freshly allocated qubits match. 
 * Compute a permutation to make clever choices across all child branches: Cost function is defined by the number of cycles that can be moved out of the branch (both at the top and at the bottom). 
 * Optimization: before computing any permutation across child branches, check when the first cycle is within the if-statement across all branches where an "outside qubit" is used, same for the bottom.
    -> If the cycles are first and last, never mind computation a permutation, if not you can stop searching for the best permutation if you have found a permutaiton that makse all instruction for the cycles before first outside use  and after last outside use the same. 
 * Last step, is to compute a final permutation of the freshly allocated qubits in the child branches to integrate them into the parent branch.
 */