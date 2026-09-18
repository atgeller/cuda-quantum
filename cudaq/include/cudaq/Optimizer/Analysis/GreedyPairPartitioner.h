/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace cudaq::opt {

/// Greedy qubit-partition analysis driven by interactions alone.
///
/// Only a multi-qubit gate constrains placement: it needs its operands in one
/// region. A single-qubit gate runs wherever its qubit already is, so here it
/// never opens, closes or merges a partition -- partitions are formed by
/// interactions, and everything else rides along with its qubit.
///
/// A qubit's ops before its first interaction are held, then pulled into the
/// partition of that interaction, so a qubit is created and prepared where it
/// is first needed rather than wherever it happened to be allocated. A qubit
/// that never interacts is flushed at the end, packed with others up to
/// \p maxQubits, so every operation still lands in some partition.
struct GreedyPairPartitioner {
  explicit GreedyPairPartitioner(mlir::Operation *op, unsigned maxQubits);

  mlir::ArrayRef<llvm::DenseSet<mlir::Operation *>> getPartitions() const {
    return partitions;
  }

private:
  void partitionBlock(mlir::Block &block, unsigned maxQubits);

  mlir::SmallVector<llvm::DenseSet<mlir::Operation *>> partitions;
};

} // namespace cudaq::opt
