#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Dialect/CFDlang/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::cfdlang;

#define DEBUG_TYPE "cfdlang-kernelize"

namespace {

MemRefType bufferType(AtomType type)
{
    return MemRefType::get(type.getShape(), type.getElementType());
}

class KernelizePass
        : public KernelizeBase<KernelizePass> {
public:
    virtual void runOnOperation() override
    {
        for (auto program : getOperation().getOps<ProgramOp>()) {
            runOnProgram(program);
        }
    }

    virtual void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<memref::MemRefDialect>();
        registry.insert<bufferization::BufferizationDialect>();
    }

private:
    void runOnProgram(ProgramOp program)
    {
        this->program = program;
        SmallVector<Type, 8> argTypes;
        for (auto input : program.getOps<InputOp>()) {
            argTypes.push_back(bufferType(input.getAtomType()));
        }
        for (auto output : program.getOps<OutputOp>()) {
            argTypes.push_back(bufferType(output.getAtomType()));
        }

        OpBuilder builder(program);
        auto kernelFn = builder.create<FuncOp>(
            program.getLoc(),
            program.getName().getValueOr("kernel"),
            FunctionType::get(
                program.getContext(),
                argTypes,
                {}
            )
        );
        kernelFn.setPublic();
        auto body = kernelFn.addEntryBlock();
        builder.setInsertionPointToStart(body);

        unsigned idx = 0;
        for (auto input : program.getOps<InputOp>()) {
            materializedValues.try_emplace(
                input.getName(),
                body->getArgument(idx++)
            );
        }
        for (auto output : program.getOps<OutputOp>()) {
            materialize(builder, output, body->getArgument(idx++));
        }

        builder.create<ReturnOp>(program.getLoc());

        program->remove();
        program->destroy();
    }

    Value getOrMaterialize(OpBuilder& builder, StringRef name)
    {
        auto find = materializedValues.find(name);
        if (find != materializedValues.end()) {
            return find->second;
        }

        auto definition = dyn_cast<DefinitionOp>(this->program.lookupSymbol(name));
        assert(definition);

        return materialize(builder, definition);
    }
    Value materialize(OpBuilder& builder, DefinitionOp definition, Value buffer = {})
    {
        if (!buffer) {
            buffer = builder.create<memref::AllocaOp>(
                definition.getLoc(),
                bufferType(definition.getAtomType())
            );
        }

        SmallVector<Type, 8> argTypes;
        SmallVector<Value, 8> args;

        for (auto eval : definition->getRegion(0).getOps<EvalOp>()) {
            argTypes.push_back(bufferType(eval.getResult().getType().cast<AtomType>()));
            args.push_back(getOrMaterialize(builder, eval.name().getLeafReference()));
        }
        argTypes.push_back(buffer.getType());
        args.push_back(buffer);

        OpBuilder outer(builder.getInsertionBlock()->getParentOp());
        auto nodeFn = outer.create<FuncOp>(
            definition.getLoc(),
            Twine("node_").concat(definition.getName()).str(),
            FunctionType::get(
                definition.getContext(),
                argTypes,
                {}
            )
        );
        nodeFn.setPrivate();

        OpBuilder nodeBuilder(builder);
        auto nodeBody = nodeFn.addEntryBlock();
        nodeBuilder.setInsertionPointToStart(nodeBody);

        BlockAndValueMapping mapping;
        unsigned idx = 0;
        for (auto& op : *(definition->getRegion(0).begin())) {
            if (auto eval = dyn_cast<EvalOp>(op)) {
                mapping.map(
                    eval.getResult(),
                    nodeBuilder.create<bufferization::ToTensorOp>(
                        eval.getLoc(),
                        nodeBody->getArgument(idx++)
                    )
                );
                continue;
            }
            if (auto yield = dyn_cast<YieldOp>(op)) {
                auto memref = nodeBuilder.create<bufferization::ToMemrefOp>(
                    yield.getLoc(),
                    bufferType(yield.atom().cast<Atom>().getType()),
                    mapping.lookup(yield.getOperand())
                );
                nodeBuilder.create<memref::CopyOp>(
                    yield.getLoc(),
                    memref,
                    nodeBody->getArgument(args.size() - 1)
                );
                break;
            }
            nodeBuilder.clone(op, mapping);
        }

        nodeBuilder.create<ReturnOp>(definition.getLoc());

        builder.create<CallOp>(
            definition.getLoc(),
            nodeFn,
            args
        );

        materializedValues.try_emplace(definition.getName(), buffer);
        return buffer;
    }

    DenseMap<StringRef, Value> materializedValues;
    ProgramOp program;
};

} // namespace <anonymous>

std::unique_ptr<OperationPass<ModuleOp>> mlir::cfdlang::createKernelizePass() {
    return std::make_unique<KernelizePass>();
}
