
/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Executor.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/RuntimeMLIR.h"
#include "common/ArgumentWrapper.h"
#include "common/RuntimeMLIRCommonImpl.h"
#include "cudaq.h"
#include "nvqpp_config.h"

#include "cudaq.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include "test_qpu.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/Base64.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"

#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <regex>
#include <sys/socket.h>
#include <sys/types.h>

namespace cudaq::test {
cudaq::sample_result sample(std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles,
                            std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            int n_samples) {
  cudaq::ExecutionContext context("sample", n_samples);
  auto &platform = get_platform();
  platform.set_exec_ctx(&context, 0);
  cudaq::altLaunchKernel("test_launch", nullptr, nullptr,
                         0, 0);

  return context.result;
}
} // namespace cudaq::test

using namespace mlir;

namespace cudaq {

/// @brief The OrcaRemoteRESTQPU is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service.
/// Moreover, this QPU handles launching kernels under the Execution Context
/// that includs sampling via synchronous client invocations.
class TestQPU : public cudaq::QPU {
protected:
  /// @brief The Pass pipeline string, configured by the
  /// QPU configuration file in the platform path.
  std::string passPipelineConfig = "canonicalize";  

  /// @brief Name of code generation target (e.g. `qir-adaptive`, `qir-base`,
  /// `qasm2`, `iqm`)
  std::string codegenTranslation = "";

  /// @brief Additional passes to run after the codegen-specific passes
  std::string postCodeGenPasses = "";

  /// @brief the platform file path, CUDAQ_INSTALL/platforms
  std::filesystem::path platformPath;

  /// @brief The name of the QPU being targeted
  std::string qpuName;
  
  /// @brief Mapping of general key-values for backend
  /// configuration.
  std::map<std::string, std::string> backendConfig;

  /// @brief Flag indicating whether we should perform the passes in a
  /// single-threaded environment, useful for debug. Similar to
  /// `-mlir-disable-threading` for `cudaq-opt`.
  bool disableMLIRthreading = false;

  /// @brief Flag indicating whether we should enable MLIR printing before and
  /// after each pass. This is similar to `-mlir-print-ir-before-all` and
  /// `-mlir-print-ir-after-all` in `cudaq-opt`.
  bool enablePrintMLIREachPass = false;

public:
  /// @brief The constructor
  TestQPU() : QPU() {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
  }

  TestQPU(TestQPU &&) = delete;

  /// @brief The destructor
  virtual ~TestQPU() = default;

  /// @brief Returns the name of the server helper.
  const std::string name() const { return "test"; }

  void enqueue(cudaq::QuantumTask&) override {}

  void setExecutionContext(cudaq::ExecutionContext*) override {}

  void resetExecutionContext() override {}

  /// @brief Helper function to get boolean environment variable
  bool getEnvBool(const char *envName, bool defaultVal = false) {
    if (auto envVal = std::getenv(envName)) {
      std::string tmp(envVal);
      std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                      [](unsigned char c) { return std::tolower(c); });
      if (tmp == "1" || tmp == "on" || tmp == "true" || tmp == "yes")
        return true;
    }
    return defaultVal;
  }

  std::tuple<ModuleOp, MLIRContext *, void *>
    extractQuakeCodeAndContext(const std::string &kernelName,
                              void *data) {
      auto contextPtr = cudaq::initializeMLIR();
      MLIRContext &context = *contextPtr.get();

      // Get the quake representation of the kernel
      auto quakeCode = cudaq::get_quake_by_name(kernelName);
      auto m_module = parseSourceString<ModuleOp>(quakeCode, &context);
      if (!m_module)
        throw std::runtime_error("module cannot be parsed");

      return std::make_tuple(m_module.release(), contextPtr.release(), data);
    }

  /// @brief Launch the kernel. Handle all pertinent
  /// modifications for the execution context.
  void launchKernel(const std::string &kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override;

  /// @brief This setTargetBackend override is in charge of reading the
  /// specific target backend configuration file (bundled as part of this
  /// CUDA-Q installation) and extract MLIR lowering pipelines and
  /// specific code generation output required by this backend (QIR/QASM2).
  void setTargetBackend(const std::string &backend) override;
};

/// @brief This setTargetBackend override is in charge of reading the
/// specific target backend configuration file (bundled as part of this
/// CUDA-Q installation) and extract MLIR lowering pipelines and
/// specific code generation output required by this backend (QIR/QASM2).
void TestQPU::setTargetBackend(const std::string &backend) {
  cudaq::info("Targeting {}.", backend);

  // First we see if the given backend has extra config params
  auto mutableBackend = backend;
  if (mutableBackend.find(";") != std::string::npos) {
    auto split = cudaq::split(mutableBackend, ';');
    mutableBackend = split[0];
    // Must be key-value pairs, therefore an even number of values here
    if ((split.size() - 1) % 2 != 0)
      throw std::runtime_error(
          "Backend config must be provided as key-value pairs: " +
          std::to_string(split.size()));

    // Add to the backend configuration map
    for (std::size_t i = 1; i < split.size(); i += 2) {
      // No need to decode trivial true/false values
      if (split[i + 1].starts_with("base64_")) {
        split[i + 1].erase(0, 7); // erase "base64_"
        std::vector<char> decoded_vec;
        if (auto err = llvm::decodeBase64(split[i + 1], decoded_vec))
          throw std::runtime_error("DecodeBase64 error");
        std::string decodedStr(decoded_vec.data(), decoded_vec.size());
        cudaq::info("Decoded {} parameter from '{}' to '{}'", split[i],
                    split[i + 1], decodedStr);
        backendConfig.insert({split[i], decodedStr});
      } else {
        backendConfig.insert({split[i], split[i + 1]});
      }
    }
  }

  // Print the IR if requested
  //printIR = getEnvBool("CUDAQ_DUMP_JIT_IR", printIR);

  // Get additional debug values
  disableMLIRthreading =
      getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", disableMLIRthreading);
  enablePrintMLIREachPass =
      getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", enablePrintMLIREachPass);

  // If the very verbose enablePrintMLIREachPass flag is set, then
  // multi-threading must be disabled.
  if (enablePrintMLIREachPass) {
    disableMLIRthreading = true;
  }

  /// Once we know the backend, we should search for the configuration file
  /// from there we can get the URL/PORT and the required MLIR pass
  /// pipeline.
  std::string fileName = mutableBackend + std::string(".config");
  auto configFilePath = platformPath / fileName;
  cudaq::info("Config file path = {}", configFilePath.string());
  std::ifstream configFile(configFilePath.string());
  std::string configContents((std::istreambuf_iterator<char>(configFile)),
                              std::istreambuf_iterator<char>());

  // Loop through the file, extract the pass pipeline and CODEGEN Type
  auto lines = cudaq::split(configContents, '\n');
  std::regex pipeline("^PLATFORM_LOWERING_CONFIG\\s*=\\s*\"(\\S+)\"");
  std::regex emissionType("^CODEGEN_EMISSION\\s*=\\s*(\\S+)");
  std::regex postCodeGen("^POST_CODEGEN_PASSES\\s*=\\s*\"(\\S+)\"");
  std::smatch match;
  for (const std::string &line : lines) {
    if (std::regex_search(line, match, pipeline)) {
      cudaq::info("Appending lowering pipeline: {}", match[1].str());
      passPipelineConfig += "," + match[1].str();
    } else if (std::regex_search(line, match, emissionType)) {
      codegenTranslation = match[1].str();
    } else if (std::regex_search(line, match, postCodeGen)) {
      cudaq::info("Adding post-codegen lowering pipeline: {}",
                  match[1].str());
      postCodeGenPasses = match[1].str();
    }
  }
  std::string allowEarlyExitSetting =
      (codegenTranslation == "qir-adaptive") ? "1" : "0";
  passPipelineConfig = std::string("cc-loop-unroll{allow-early-exit=") +
                        allowEarlyExitSetting + "}," + passPipelineConfig;

  auto disableQM = backendConfig.find("disable_qubit_mapping");
  if (disableQM != backendConfig.end() && disableQM->second == "true") {
    // Replace the qubit-mapping{device=<>} with
    // qubit-mapping{device=bypass} to effectively disable the qubit-mapping
    // pass. Use $1 - $4 to make sure any other pass options are left
    // untouched.
    std::regex qubitMapping(
        "(.*)qubit-mapping\\{(.*)device=[^,\\}]+(.*)\\}(.*)");
    std::string replacement("$1qubit-mapping{$2device=bypass$3}$4");
    passPipelineConfig =
        std::regex_replace(passPipelineConfig, qubitMapping, replacement);
    cudaq::info("disable_qubit_mapping option found, so updated lowering "
                "pipeline to {}",
                passPipelineConfig);
  }

  // Set the qpu name
  qpuName = mutableBackend;
}

/// @brief Launch the kernel.
void TestQPU::launchKernel(const std::string &kernelName,
                                     void (*kernelFunc)(void *), void *args,
                                     std::uint64_t voidStarSize,
                                     std::uint64_t resultOffset) {
  cudaq::info("launching Test remote rest kernel ({})", kernelName);

  // TODO future iterations of this should support non-void return types.
  if (!executionContext)
    throw std::runtime_error("Remote rest execution can only be performed "
                             "via cudaq::sample() or cudaq::observe().");

  auto [m_module, contextPtr, updatedArgs] =
        extractQuakeCodeAndContext(kernelName, args);

  mlir::MLIRContext &context = *contextPtr;

  // Extract the kernel name
  auto func = m_module.lookupSymbol<mlir::func::FuncOp>(
      std::string("__nvqpp__mlirgen__") + kernelName);

  // Create a new Module to clone the function into
  auto location = mlir::FileLineColLoc::get(&context, "<builder>", 1, 1);
  mlir::ImplicitLocOpBuilder builder(location, &context);

  // FIXME this should be added to the builder.
  if (!func->hasAttr(cudaq::entryPointAttrName))
    func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
  auto moduleOp = builder.create<mlir::ModuleOp>();
  moduleOp.push_back(func.clone());
  moduleOp->setAttrs(m_module->getAttrDictionary());

  // Lambda to apply a specific pipeline to the given ModuleOp
  auto runPassPipeline = [&](const std::string &pipeline,
                              mlir::ModuleOp moduleOpIn) {
    mlir::PassManager pm(&context);
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    cudaq::info("Pass pipeline for {} = {}", kernelName, pipeline);
    if (failed(parsePassPipeline(pipeline, pm, os)))
      throw std::runtime_error(
          "Failed to add passes to pipeline (" + errMsg +
          ").");
    //if (disableMLIRthreading || enablePrintMLIREachPass)
    // Disable by default for now
    moduleOpIn.getContext()->disableMultithreading();
    //if (enablePrintMLIREachPass)
    // Enable by default for now
    pm.enableIRPrinting();
    if (failed(pm.run(moduleOpIn)))
      throw std::runtime_error("Quake lowering failed.");
  };

  if (updatedArgs) {
    cudaq::info("Run Quake Synth.\n");
    mlir::PassManager pm(&context);
    pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, updatedArgs));
    //if (disableMLIRthreading || enablePrintMLIREachPass)
    // Disable by default for now
    moduleOp.getContext()->disableMultithreading();
    //if (enablePrintMLIREachPass)
    // Enable by default for now
    pm.enableIRPrinting();
    if (failed(pm.run(moduleOp)))
      throw std::runtime_error("Could not successfully apply quake-synth.");
  }

  // Run the config-specified pass pipeline
  runPassPipeline(passPipelineConfig, moduleOp);
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, TestQPU, test)
}