//===- Inference.td - Inference method utilities -----------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This file provides some common definitions for inference methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONCPETS_INFERENCE
#define MLIR_CONCEPTS_INFERENCE

include "mlir/IR/OpBase.td"

// Inference method utilities.
def InferenceMethod {
    dag arguments = (ins
        "::mlir::MLIRContext*":$context,
        "::std::optional<::mlir::Location>":$location,
        "::mlir::ValueRange":$operands,
        "::mlir::DictionaryAttr":$attributes,
        "::mlir::OpaqueProperties":$properties,
        "::mlir::RegionRange":$regions
    );
    code forwardArguments = [{
        context, location, operands, attributes, properties, regions
    }];
}

// Functor for substitution within an inference method.
class InferenceMethodImpl<code bodyCode> {
    code body = !subst("$_args", InferenceMethod.forwardArguments, bodyCode);
}

#endif
