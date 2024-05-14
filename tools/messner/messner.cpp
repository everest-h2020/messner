/// Main entry point for the messner CLI facade.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Target/EKL/Import.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::ekl;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input file>"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o",
    llvm::cl::desc("Output filename"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

/*
// Prepare the pass manager, applying command-line and reproducer options.
  PassManager pm(op.get()->getName(), PassManager::Nesting::Implicit);
  pm.enableVerifier(config.shouldVerifyPasses());
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableTiming(timing);
  if (config.shouldRunReproducer() && failed(reproOptions.apply(pm)))
    return failure();
  if (failed(config.setupPassPipeline(pm)))
    return failure();

  // Run the pipeline.
  if (failed(pm.run(*op)))
    return failure();
*/

OwningOpRef<ModuleOp> runOnInput(OwningOpRef<ProgramOp> input)
{
    // Create the result ModuleOp and put the program in it.
    OpBuilder builder(input->getContext());
    OwningOpRef<ModuleOp> result = builder.create<ModuleOp>(input->getLoc());
    builder.setInsertionPointToStart(result->getBody());
    builder.insert(input.release());

    // Create and set up the pass manager.
    PassManager passManager(
        result->getOperation()->getName(),
        PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(passManager))) return {};

    // TODO: Populate the pass pipeline.
    passManager.nest(ProgramOp::getOperationName())
        .addNestedPass<KernelOp>(createLowerPass());

    // Run the pass manager on the module.
    if (failed(passManager.run(result->getOperation()))) return {};

    // Verify the result before proceeding.
    if (failed(verify(*result))) return {};
    return result;
}

LogicalResult runOnInput(
    const DialectRegistry &registry,
    std::unique_ptr<llvm::MemoryBuffer> input,
    llvm::raw_ostream &output)
{
    // Create and set up the llvm::SourceMgr.
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(input), llvm::SMLoc{});

    // Create and set up the MLIRContext.
    MLIRContext context(registry);
    context.printOpOnDiagnostic(false);
    context.loadAllAvailableDialects();

    // Attach our rich diagnostic handler.
    DiagHandler diagHandler(&context, sourceMgr);

    // Import, verify and type check the input program.
    auto program = importAndTypeCheck(&context, sourceMgr);
    if (!program) return failure();

    // Run the pass pipeline to produce the result module.
    auto module = runOnInput(std::move(program));
    if (!module) return failure();

    // The default constructor will use the printer flags from the CLI.
    AsmState state(*module);
    module->print(output, state);
    return success();
}

LogicalResult runOnInput(const DialectRegistry &registry)
{
    // Display a warning when interactive input is used.
    if (inputFilename == "-"
        && llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
        llvm::errs()
            << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
               "interrupt)\n";

    // Open the input and output files.
    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
        llvm::errs() << errorMessage << "\n";
        return failure();
    }
    auto output = openOutputFile(outputFilename, &errorMessage);
    if (!output) {
        llvm::errs() << errorMessage << "\n";
        return failure();
    }

    // Run the compiler.
    if (failed(runOnInput(registry, std::move(file), output->os())))
        return failure();

    // Only keep the output file on disk on success.
    output->keep();
    return success();
}

int main(int argc, char *argv[])
{
    llvm::InitLLVM init(argc, argv);

    // Handle command-line arguments.
    DialectRegistry registry;
    MlirOptMainConfig::registerCLOptions(registry);
    registerAsmPrinterCLOptions();
    registerMLIRContextCLOptions();
    registerPassManagerCLOptions();
    llvm::cl::ParseCommandLineOptions(argc, argv);
    MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();

    // Populate the dialect registry.
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<EKLDialect>();

    // Invoke the compiler.
    return asMainReturnCode(runOnInput(registry));
}
