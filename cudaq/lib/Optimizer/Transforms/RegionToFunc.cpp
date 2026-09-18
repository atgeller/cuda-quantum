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
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_REGIONTOFUNC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

static constexpr const char *regionAttrName = "region";

namespace {
class RegionToFuncPass
    : public cudaq::opt::impl::RegionToFuncBase<RegionToFuncPass> {
  using Base = cudaq::opt::impl::RegionToFuncBase<RegionToFuncPass>;

public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SymbolTable symbolTable(mod);

    SmallVector<func::FuncOp> parents(mod.getOps<func::FuncOp>());
    for (func::FuncOp parent : parents) {
      SmallVector<cudaq::cc::CreateLambdaOp> lambdas;
      parent.walk([&](cudaq::cc::CreateLambdaOp lambda) {
        if (lambda->hasAttr(regionAttrName))
          lambdas.push_back(lambda);
      });
      for (auto [idx, lambda] : llvm::enumerate(lambdas))
        if (failed(outline(lambda, idx, parent, symbolTable)))
          return signalPassFailure();
    }
  }

private:
  /// Rewrite one region-tagged subcircuit lambda into a `func.func` carrying the
  /// same `{region}` attribute, and its `cc.call_callable` into a `func.call`.
  /// The body moves across untouched, so the entry `quake.move` ops that place
  /// wires on the region stay with the code they describe.
  LogicalResult outline(cudaq::cc::CreateLambdaOp lambda, size_t idx,
                        func::FuncOp parent, SymbolTable &symbolTable) {
    auto call = soleCallCallable(lambda);
    if (!call)
      return lambda.emitOpError(
          "region subcircuit must have exactly one cc.call_callable user; "
          "run region-to-func directly after add-region-moves");

    auto regionAttr = lambda->getAttrOfType<FlatSymbolRefAttr>(regionAttrName);
    auto callableTy =
        cast<cudaq::cc::CallableType>(lambda.getSignature().getType());
    FunctionType fnTy = callableTy.getSignature();

    OpBuilder builder(parent);
    auto name = (parent.getName() + "." + regionAttr.getValue() + "." +
                 Twine(idx))
                    .str();
    auto subFunc = func::FuncOp::create(builder, lambda.getLoc(), name, fnTy);
    subFunc.setPrivate();
    subFunc->setAttr(regionAttrName, regionAttr);
    symbolTable.insert(subFunc);

    subFunc.getBody().takeBody(lambda.getInitRegion());
    // `cc.return` is the lambda terminator; a func needs `func.return`.
    auto ret = cast<cudaq::cc::ReturnOp>(subFunc.getBody().back().back());
    builder.setInsertionPoint(ret);
    func::ReturnOp::create(builder, ret.getLoc(), ret.getOperands());
    ret.erase();

    builder.setInsertionPoint(call);
    auto newCall = func::CallOp::create(builder, call.getLoc(), subFunc,
                                        call.getArgs());
    call.replaceAllUsesWith(newCall.getResults());
    call.erase();
    lambda.erase();
    return success();
  }

  static cudaq::cc::CallCallableOp
  soleCallCallable(cudaq::cc::CreateLambdaOp lambda) {
    cudaq::cc::CallCallableOp found;
    for (Operation *user : lambda->getUsers()) {
      auto call = dyn_cast<cudaq::cc::CallCallableOp>(user);
      if (!call || found)
        return {};
      found = call;
    }
    return found;
  }
};
} // namespace
