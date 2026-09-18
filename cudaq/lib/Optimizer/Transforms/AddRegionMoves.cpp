/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ADDREGIONMOVES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

static bool isWire(Value v) {
  return isa<cudaq::quake::WireType>(v.getType());
}

// Return the assigned region name for a lambda, or "" if not annotated.
static StringRef getRegionAttr(cudaq::cc::CreateLambdaOp lambda) {
  auto attr = lambda->getAttrOfType<FlatSymbolRefAttr>(
      StringAttr::get(lambda.getContext(), "region"));
  return attr ? attr.getValue() : StringRef{};
}

// At the entry of each subcircuit body, move wire block arguments into the
// region. At the exit, move returned wires out before cc.return.
static void addIntraBodyMoves(cudaq::cc::CreateLambdaOp lambda) {
  StringRef regionName = getRegionAttr(lambda);
  if (regionName.empty())
    return;

  MLIRContext *ctx = lambda.getContext();
  auto wireTy = cudaq::quake::WireType::get(ctx);
  auto symRef = FlatSymbolRefAttr::get(ctx, regionName);
  Location loc = lambda.getLoc();
  OpBuilder builder(ctx);
  Block &body = lambda.getBody().front();

  // Entry moves: wire arg → region slot (slot indices assigned in arg order).
  builder.setInsertionPointToStart(&body);
  int32_t slot = 0;
  for (BlockArgument arg : body.getArguments()) {
    if (!isWire(arg))
      continue;
    auto slotAttr = builder.getI32IntegerAttr(slot++);
    auto move = builder.create<cudaq::quake::MoveOp>(
        loc, wireTy, arg, symRef, slotAttr, builder.getStringAttr("wires"));
    arg.replaceUsesWithIf(move.getResult(), [&](OpOperand &use) {
      return use.getOwner() != move.getOperation();
    });
  }

  // Exit moves: a wire that leaves the subcircuit still alive is handed to the
  // region's port, which is what frees the compute wire it was holding. A wire
  // returned to the wire set inside the body needs nothing -- it is finished,
  // and its wire comes free with it. Without this the IR says only where a
  // qubit arrives and never that it left, so a region can never be vacated to
  // make room for the next arrival.
  Operation *terminator = body.getTerminator();
  if (isa<cudaq::cc::ReturnOp>(terminator)) {
    builder.setInsertionPoint(terminator);
    int32_t port = 0;
    for (OpOperand &use : terminator->getOpOperands()) {
      if (!isWire(use.get()))
        continue;
      auto move = builder.create<cudaq::quake::MoveOp>(
          loc, wireTy, use.get(), symRef, builder.getI32IntegerAttr(port++),
          builder.getStringAttr("out"));
      use.set(move.getResult());
    }
  }
}

// In the enclosing block, insert a quake.move before each cc.call_callable
// carrying every wire argument onto the destination region's in-port, in
// argument order. A region is entered only through its in-port, so this fires
// even when the argument is coming back to the region it just left: the
// boundary is what costs, not the distance.
static void addInterSubcircuitMoves(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  auto wireTy = cudaq::quake::WireType::get(ctx);
  OpBuilder builder(ctx);

  func.walk([&](cudaq::cc::CallCallableOp call) {
    auto destLambda =
        call.getCallee().getDefiningOp<cudaq::cc::CreateLambdaOp>();
    if (!destLambda)
      return;
    StringRef destRegion = getRegionAttr(destLambda);
    if (destRegion.empty())
      return;

    builder.setInsertionPoint(call);
    for (auto [i, arg] : llvm::enumerate(call.getArgs())) {
      if (!isWire(arg))
        continue;

      StringRef srcRegion;
      if (auto srcCall = arg.getDefiningOp<cudaq::cc::CallCallableOp>()) {
        auto srcLambda =
            srcCall.getCallee().getDefiningOp<cudaq::cc::CreateLambdaOp>();
        if (srcLambda)
          srcRegion = getRegionAttr(srcLambda);
      }

      (void)srcRegion;
      auto symRef = FlatSymbolRefAttr::get(ctx, destRegion);
      auto move = builder.create<cudaq::quake::MoveOp>(
          call.getLoc(), wireTy, arg, symRef,
          builder.getI32IntegerAttr(static_cast<int32_t>(i)),
          builder.getStringAttr("in"));
      // Operand 0 is the callee; args start at operand 1.
      call->setOperand(1 + i, move.getResult());
    }
  });
}

// A wire whose only remaining use is being returned to the wire set is
// finished, and its wire comes free with it -- so it does not vacate onto a
// port first. This is the mirror of a qubit's first use, which is placed on a
// compute wire outright rather than arriving through one.
static void dropExitMovesForFinishedWires(func::FuncOp func) {
  func.walk([&](cudaq::cc::CallCallableOp call) {
    auto lambda = call.getCallee().getDefiningOp<cudaq::cc::CreateLambdaOp>();
    if (!lambda || lambda.getBody().empty())
      return;
    Operation *terminator = lambda.getBody().front().getTerminator();
    for (auto [i, result] : llvm::enumerate(call.getResults())) {
      if (!isWire(result))
        continue;
      // A wire with no uses at all is finished too -- nothing downstream ever
      // asks where it is, so vacating onto a port buys nothing.
      if (!llvm::all_of(result.getUsers(), [](Operation *user) {
            return isa<cudaq::quake::ReturnWireOp, cudaq::quake::SinkOp>(user);
          }))
        continue;
      if (i >= terminator->getNumOperands())
        continue;
      auto move =
          terminator->getOperand(i).getDefiningOp<cudaq::quake::MoveOp>();
      if (!move || move.getDestKind() != "out")
        continue;
      terminator->setOperand(i, move.getWire());
      if (move.getResult().use_empty())
        move.erase();
    }
  });
}

class AddRegionMovesPass
    : public cudaq::opt::impl::AddRegionMovesBase<AddRegionMovesPass> {
  using Base = cudaq::opt::impl::AddRegionMovesBase<AddRegionMovesPass>;

public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<cudaq::cc::CreateLambdaOp> lambdas;
    func.walk([&](cudaq::cc::CreateLambdaOp lambda) {
      lambdas.push_back(lambda);
    });
    if (lambdas.empty())
      return;

    for (auto lambda : lambdas)
      addIntraBodyMoves(lambda);

    addInterSubcircuitMoves(func);
    dropExitMovesForFinishedWires(func);
  }
};
} // namespace

namespace cudaq::opt {
std::unique_ptr<mlir::Pass> createAddRegionMovesPass() {
  return std::make_unique<AddRegionMovesPass>();
}
} // namespace cudaq::opt
