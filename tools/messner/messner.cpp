/// Main entry point for the messner CLI facade.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToFunc/EKLToFunc.h"
#include "messner/Conversion/EKLToLinalg/EKLToLinalg.h"
#include "messner/Conversion/EKLToStandard/EKLToStandard.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Target/EKL/Import.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <llvm/IR/LLVMContext.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>

using namespace mlir;
using namespace mlir::ekl;

namespace {

class CopyElisionPass : public OperationPass<> {
public:
    CopyElisionPass() : OperationPass<>(TypeID::get<CopyElisionPass>()) {}
    CopyElisionPass(const CopyElisionPass &other) : OperationPass<>(other) {}
    CopyElisionPass &operator=(const CopyElisionPass &) = delete;
    CopyElisionPass(CopyElisionPass &&)                 = delete;
    CopyElisionPass &operator=(CopyElisionPass &&)      = delete;
    ~CopyElisionPass()                                  = default;

    static constexpr ::llvm::StringLiteral getArgumentName()
    {
        return ::llvm::StringLiteral("ekl-copy-elision");
    }
    ::llvm::StringRef getArgument() const override
    {
        return "ekl-copy-elision";
    }
    ::llvm::StringRef getDescription() const override { return ""; }
    static constexpr ::llvm::StringLiteral getPassName()
    {
        return ::llvm::StringLiteral("CopyElision");
    }
    ::llvm::StringRef getName() const override { return "CopyElision"; }

    static bool classof(const Pass *pass)
    {
        return pass->getTypeID() == TypeID::get<CopyElisionPass>();
    }

    std::unique_ptr<Pass> clonePass() const override
    {
        return std::make_unique<CopyElisionPass>(*this);
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<ekl::EKLDialect>();
        registry.insert<linalg::LinalgDialect>();
        registry.insert<memref::MemRefDialect>();
    }

    void runOnOperation() override
    {
        getOperation()->walk([](memref::CopyOp copy) {
            const auto src = llvm::dyn_cast<OpResult>(copy.getSource());
            const auto dst = llvm::dyn_cast<BlockArgument>(copy.getTarget());
            if (!src || !dst) return WalkResult::advance();
            auto map =
                llvm::dyn_cast_if_present<linalg::MapOp>(copy->getPrevNode());
            if (!map) return WalkResult::advance();
            if (map->getOperand(0) != src) return WalkResult::advance();
            const auto alloc = llvm::dyn_cast<memref::AllocOp>(src.getOwner());
            if (!alloc
                || !llvm::isa<func::FuncOp>(dst.getOwner()->getParentOp()))
                return WalkResult::advance();

            IRRewriter rewriter(copy);
            map->setOperand(0, dst);
            rewriter.eraseOp(copy);
            if (alloc->use_empty()) rewriter.eraseOp(alloc);
            return WalkResult::skip();
        });
    }

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyElisionPass)
};

} // namespace

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input file>"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o",
    llvm::cl::desc("Output filename"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

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

    auto &program = passManager.nest<ProgramOp>();
    auto &kernel  = program.nest<KernelOp>();
    // -ekl-lower -ekl-decay-number -ekl-homogenize -ekl-implement
    kernel.addPass(createLowerPass());
    kernel.addPass(createDecayNumberPass());
    kernel.addPass(createHomogenizePass());
    kernel.addPass(createImplementPass());
    // -cse -canonicalize
    kernel.addPass(createCSEPass());
    kernel.addPass(createCanonicalizerPass());

    // -ekl-to-func -ekl-to-linalg -ekl-to-std
    passManager.addPass(messner::createConvertEKLToFuncPass());
    passManager.addPass(messner::createConvertEKLToLinalgPass());
    passManager.addPass(messner::createConvertEKLToStandardPass());
    // -reconcile-unrealized-casts -cse -canonicalize
    passManager.addPass(createReconcileUnrealizedCastsPass());
    passManager.addPass(createCSEPass());
    passManager.addPass(createCanonicalizerPass());
    // -eliminate-empty-tensors
    passManager.addPass(bufferization::createEmptyTensorEliminationPass());
    // -one-shot-bufferize
    passManager.addPass(bufferization::createOneShotBufferizePass());
    // Magic copy elision fix.
    passManager.addPass(std::make_unique<CopyElisionPass>());
    // -convert-linalg-to-loops
    passManager.addPass(createConvertLinalgToLoopsPass());
    // -buffer-loop-hoisting -buffer-hoosting -buffer-deallocation
    passManager.addPass(bufferization::createBufferLoopHoistingPass());
    passManager.addPass(bufferization::createBufferHoistingPass());
    passManager.addPass(bufferization::createBufferDeallocationPass());
    // -expand-strided-metadata -finalize-memref-to-llvm -lower-affine
    passManager.addPass(memref::createExpandStridedMetadataPass());
    passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
    passManager.addPass(createLowerAffinePass());
    // -convert-scf-to-cf
    passManager.addPass(createConvertSCFToCFPass());
    // -convert-func-to-llvm="use-bare-ptr-memref-call-conv=1"
    ConvertFuncToLLVMPassOptions funcToLLVMOptions{true, 64};
    passManager.addPass(createConvertFuncToLLVMPass(funcToLLVMOptions));
    // -reconcile-unrealized-casts -cse -canonicalize
    passManager.addPass(createReconcileUnrealizedCastsPass());
    passManager.addPass(createCSEPass());
    passManager.addPass(createCanonicalizerPass());
    // -convert-to-llvm
    passManager.addPass(createConvertToLLVMPass());
    // -reconcile-unrealized-casts
    passManager.addPass(createReconcileUnrealizedCastsPass());

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

    // Translate to LLVMIR.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());
    llvm::LLVMContext llvmCtx;
    auto llvmModule = translateModuleToLLVMIR(*module, llvmCtx);
    if (!llvmModule) return failure();

    if (inputFilename == "-")
        llvmModule->setSourceFileName("EKLProgram");
    else
        llvmModule->setSourceFileName(inputFilename);

    // Run an optimization pipeline over the module.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) return failure();
    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) return failure();
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
        llvmModule.get(),
        tmOrError.get().get());
    auto optPipeline = mlir::makeOptimizingTransformer(
        3,
        /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) return failure();

    // Print the LLVMIR.
    llvmModule->print(output, nullptr);
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
    registerAllDialects(registry);
    registry.insert<EKLDialect>();

    // Register all conversions to LLVM extensions.
    arith::registerConvertArithToLLVMInterface(registry);
    registerConvertComplexToLLVMInterface(registry);
    cf::registerConvertControlFlowToLLVMInterface(registry);
    func::registerAllExtensions(registry);
    tensor::registerAllExtensions(registry);
    registerConvertFuncToLLVMInterface(registry);
    index::registerConvertIndexToLLVMInterface(registry);
    registerConvertMathToLLVMInterface(registry);
    registerConvertMemRefToLLVMInterface(registry);
    registerConvertNVVMToLLVMInterface(registry);
    registerConvertOpenMPToLLVMInterface(registry);
    ub::registerConvertUBToLLVMInterface(registry);

    // Invoke the compiler.
    return asMainReturnCode(runOnInput(registry));
}