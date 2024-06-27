/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_ASSIGNIDS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
inline bool isMeasureOp(Operation *op) {
  return dyn_cast<quake::MxOp>(*op) || dyn_cast<quake::MyOp>(*op) ||
         dyn_cast<quake::MzOp>(*op);
}

inline bool isBeginOp(Operation *op) {
  return dyn_cast<quake::UnwrapOp>(*op) || dyn_cast<quake::ExtractRefOp>(*op) ||
         dyn_cast<quake::NullWireOp>(*op);
}

inline bool isEndOp(Operation *op) {
  return dyn_cast<quake::DeallocOp>(*op) || dyn_cast<quake::SinkOp>(*op);
}

class NullWirePat : public OpRewritePattern<quake::NullWireOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  using Base = OpRewritePattern<quake::NullWireOp>;

  unsigned *counter;

  NullWirePat(MLIRContext *context, unsigned *c)
      : OpRewritePattern<quake::NullWireOp>(context), counter(c) {}

  LogicalResult matchAndRewrite(quake::NullWireOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc->hasAttr("qid"))
      return failure();

    auto qid = (*counter)++;

    rewriter.startRootUpdate(alloc);
    alloc->setAttr("qid", rewriter.getUI32IntegerAttr(qid));
    rewriter.finalizeRootUpdate(alloc);

    return success();
  }
};

std::optional<uint> findQid(Value v) {
  if (!isa<quake::WireType>(v.getType()))
    return std::optional<uint>();

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto block = arg.getParentBlock();
    // Look up operands from all branch instructions that can jump
    // to the parent block and recursively visit them
    for (auto predecessor : block->getPredecessors()) {
      if (auto branch =
              dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
        unsigned numSuccs = branch->getNumSuccessors();
        for (unsigned i = 0; i < numSuccs; ++i) {
          if (block && branch->getSuccessor(i) != block)
            continue;
          auto brArgs = branch.getSuccessorOperands(i).getForwardedOperands();
          auto operand = brArgs[arg.getArgNumber()];
          auto qid = findQid(operand);
          if (qid)
            return qid;
        }
      }
    }
  }

  auto defop = v.getDefiningOp();
  if (isBeginOp(defop)) {
    uint qid = defop->getAttr("qid").cast<IntegerAttr>().getUInt();
    return std::optional<uint>(qid);
  }

  if (isMeasureOp(defop))
    return findQid(defop->getOperand(0));

  // Figure out matching operand
  size_t i = 0;
  for (; i < defop->getNumResults(); i++)
    if (defop->getResult(i) == v)
      break;

  return findQid(defop->getOperand(i));
}

class SinkOpPat : public OpRewritePattern<quake::SinkOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  using Base = OpRewritePattern<quake::SinkOp>;

  SinkOpPat(MLIRContext *context) : OpRewritePattern<quake::SinkOp>(context) {}

  LogicalResult matchAndRewrite(quake::SinkOp release,
                                PatternRewriter &rewriter) const override {
    if (release->hasAttr("qid"))
      return failure();

    std::optional<uint> qid = findQid(release.getOperand());

    rewriter.startRootUpdate(release);
    release->setAttr("qid", rewriter.getUI32IntegerAttr(qid.value()));
    rewriter.finalizeRootUpdate(release);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct AssignIDsPass : public cudaq::opt::impl::AssignIDsBase<AssignIDsPass> {
  using AssignIDsBase::AssignIDsBase;

  void runOnOperation() override { assign(); }

  void assign() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    unsigned x = 0;
    patterns.insert<NullWirePat>(ctx, &x);
    patterns.insert<SinkOpPat>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::NullWireOp>(
        [&](quake::NullWireOp alloc) { return alloc->hasAttr("qid"); });
    target.addDynamicallyLegalOp<quake::SinkOp>(
        [&](quake::SinkOp sink) { return sink->hasAttr("qid"); });
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      func.emitOpError("factoring quantum allocations failed");
      signalPassFailure();
    }
  }
};

} // namespace