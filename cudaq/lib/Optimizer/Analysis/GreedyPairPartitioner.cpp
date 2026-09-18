/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/GreedyPairPartitioner.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include <list>

using namespace mlir;

static bool isQubitValue(Value v) {
  return isa<cudaq::quake::WireType>(v.getType());
}

static bool isSource(Operation &op) {
  return isa<cudaq::quake::NullWireOp, cudaq::quake::BorrowWireOp>(op);
}

/// How many qubit timelines an op consumes. Two or more is an interaction.
static unsigned qubitInputCount(Operation &op) {
  unsigned n = 0;
  for (Value w : op.getOperands())
    if (isQubitValue(w))
      ++n;
  return n;
}

cudaq::opt::GreedyPairPartitioner::GreedyPairPartitioner(Operation *op,
                                                         unsigned maxQubits) {
  auto func = dyn_cast<func::FuncOp>(op);
  if (!func)
    return;
  for (Block &block : func.getBody())
    partitionBlock(block, maxQubits);
}

void cudaq::opt::GreedyPairPartitioner::partitionBlock(Block &block,
                                                       unsigned maxQubits) {
  struct Part {
    SmallVector<Operation *> ops;
    DenseSet<Value> liveWires;
    unsigned qubitCount = 0;
  };

  // std::list gives stable addresses, which owner relies on.
  std::list<Part> open;
  DenseMap<Value, Part *> owner; // live wire -> the partition holding it
  // A timeline's ops that no partition has claimed yet, keyed by its current
  // wire value. Held until an interaction needs the qubit.
  DenseMap<Value, SmallVector<Operation *>> held;
  // Chains that ended without producing a wire -- a measurement or a
  // return_wire finishes the timeline, so nothing will extend them.
  SmallVector<SmallVector<Operation *>> ended;

  auto flush = [&](Part *P) {
    for (Value w : P->liveWires)
      owner.erase(w);
    if (!P->ops.empty())
      partitions.emplace_back(P->ops.begin(), P->ops.end());
    open.erase(std::find_if(open.begin(), open.end(),
                            [P](const Part &x) { return &x == P; }));
  };

  // Track a wire's ops through P: operands are consumed, results stay live.
  auto claim = [&](Part &P, Operation *op) {
    for (Value w : op->getOperands())
      if (isQubitValue(w)) {
        P.liveWires.erase(w);
        owner.erase(w);
      }
    for (Value w : op->getResults())
      if (isQubitValue(w)) {
        P.liveWires.insert(w);
        owner[w] = &P;
      }
    P.ops.push_back(op);
  };

  // Move a wire's held ops into P, in the order they appeared.
  auto claimHeld = [&](Part &P, Value wire) {
    auto it = held.find(wire);
    if (it == held.end())
      return;
    SmallVector<Operation *> chain = std::move(it->second);
    held.erase(it);
    for (Operation *o : chain)
      claim(P, o);
  };

  for (Operation &op : block) {
    bool hasQubits = llvm::any_of(op.getOperands(), isQubitValue) ||
                     llvm::any_of(op.getResults(), isQubitValue);
    if (!hasQubits)
      continue;
    if (qubitInputCount(op) > maxQubits)
      continue;

    // A source starts a timeline; hold it for whoever first interacts with it.
    if (isSource(op)) {
      for (Value w : op.getResults())
        if (isQubitValue(w))
          held[w] = {&op};
      continue;
    }

    if (qubitInputCount(op) < 2) {
      // Not an interaction: it runs wherever its qubit already is, so it never
      // decides anything. Either it joins the partition holding the wire, or
      // the wire is still held and it extends that chain.
      Value in;
      for (Value w : op.getOperands())
        if (isQubitValue(w)) {
          in = w;
          break;
        }
      if (!in)
        continue;
      auto it = owner.find(in);
      if (it != owner.end()) {
        claim(*it->second, &op);
        continue;
      }
      SmallVector<Operation *> chain;
      auto heldIt = held.find(in);
      if (heldIt != held.end()) {
        chain = std::move(heldIt->second);
        held.erase(heldIt);
      }
      chain.push_back(&op);
      Value out;
      for (Value w : op.getResults())
        if (isQubitValue(w)) {
          out = w;
          break;
        }
      if (out)
        held[out] = std::move(chain);
      else
        ended.push_back(std::move(chain));
      continue;
    }

    // An interaction: this is what forms partitions.
    SmallPtrSet<Part *, 4> touched;
    unsigned extQubits = 0;
    for (Value w : op.getOperands()) {
      if (!isQubitValue(w))
        continue;
      auto it = owner.find(w);
      if (it != owner.end())
        touched.insert(it->second);
      else
        ++extQubits; // a held timeline, entering now
    }

    unsigned merged = extQubits;
    for (Part *P : touched)
      merged += P->qubitCount;

    Part *target = nullptr;
    if (touched.empty()) {
      for (Part &P : open)
        if (P.qubitCount + extQubits <= maxQubits) {
          target = &P;
          break;
        }
    } else if (merged <= maxQubits) {
      // Fold the touched partitions together; the interaction needs its
      // operands in one region, so they cannot stay apart.
      target = *touched.begin();
      for (Part *other : llvm::drop_begin(touched)) {
        for (Operation *o : other->ops)
          target->ops.push_back(o);
        for (Value w : other->liveWires) {
          target->liveWires.insert(w);
          owner[w] = target;
        }
        target->qubitCount += other->qubitCount;
        open.erase(std::find_if(open.begin(), open.end(),
                                [other](const Part &x) { return &x == other; }));
      }
    } else {
      // Close only what has to close -- largest first -- and let the survivors
      // absorb this interaction. Committing every partition it touches would
      // scatter qubits that could have stayed together.
      DenseMap<Part *, unsigned> ownedInputs;
      for (Value w : op.getOperands()) {
        if (!isQubitValue(w))
          continue;
        auto it = owner.find(w);
        if (it != owner.end() && touched.count(it->second))
          ownedInputs[it->second]++;
      }

      SmallVector<Part *> sorted(touched.begin(), touched.end());
      llvm::sort(sorted, [](Part *a, Part *b) {
        return a->qubitCount > b->qubitCount;
      });

      // Closing a partition turns the inputs it owned into external qubits.
      SmallVector<Part *> toFlush;
      for (Part *P : sorted) {
        if (merged <= maxQubits)
          break;
        unsigned owned = ownedInputs.lookup(P);
        merged = merged - P->qubitCount + owned;
        extQubits += owned;
        toFlush.push_back(P);
      }
      SmallPtrSet<Part *, 4> flushed(toFlush.begin(), toFlush.end());
      for (Part *P : toFlush)
        flush(P);

      SmallVector<Part *> survivors;
      for (Part *P : touched)
        if (!flushed.count(P))
          survivors.push_back(P);

      if (!survivors.empty()) {
        target = survivors[0];
        for (Part *other : llvm::drop_begin(survivors)) {
          for (Operation *o : other->ops)
            target->ops.push_back(o);
          for (Value w : other->liveWires) {
            target->liveWires.insert(w);
            owner[w] = target;
          }
          target->qubitCount += other->qubitCount;
          open.erase(
              std::find_if(open.begin(), open.end(),
                           [other](const Part &x) { return &x == other; }));
        }
      } else {
        extQubits = qubitInputCount(op);
      }
    }

    if (!target) {
      open.emplace_back();
      target = &open.back();
    }
    for (Value w : op.getOperands())
      if (isQubitValue(w))
        claimHeld(*target, w);
    target->qubitCount += extQubits;
    claim(*target, &op);
  }

  // Qubits that never interacted: pack their held chains so every op still
  // lands in a partition.
  SmallVector<SmallVector<Operation *>> orphans = std::move(ended);
  for (auto &entry : held)
    if (!entry.second.empty())
      orphans.push_back(entry.second);
  // Deterministic order: by where the chain starts in the block.
  llvm::sort(orphans, [](const SmallVector<Operation *> &a,
                         const SmallVector<Operation *> &b) {
    return a.front()->isBeforeInBlock(b.front());
  });
  for (auto &chain : orphans) {
    if (open.empty() || open.back().qubitCount >= maxQubits)
      open.emplace_back();
    Part &P = open.back();
    ++P.qubitCount;
    for (Operation *o : chain)
      claim(P, o);
  }

  SmallVector<Part *> remaining;
  for (Part &P : open)
    remaining.push_back(&P);
  for (Part *P : remaining)
    flush(P);
}
