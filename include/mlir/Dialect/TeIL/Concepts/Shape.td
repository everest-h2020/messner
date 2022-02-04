//===- Shape.td - TeIL Shape concept -----------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the TeIL shape concept.
//
//===----------------------------------------------------------------------===//

#ifndef TEIL_CONCEPTS_SHAPE
#define TEIL_CONCEPTS_SHAPE

include "mlir/Concepts/Concepts.td"

def TeIL_DimSize
        : NativeTypeConcept<"::mlir::teil::DimSizeType", "dimension size">;

def TeIL_DimSizeType
        : ConstrainedTypeAttr<TeIL_DimSize>;

def TeIL_DimSizeAttr
        : NativeAttrConcept<"::mlir::teil::DimSizeAttr", "dimension size"> {
    let valueType = TeIL_DimSize;
    let constBuilderCall = [{
        ::mlir::teil::DimSizeAttr::get($_builder.getContext(), $0)
    }];
}

def TeIL_DimSizeArrayAttr
        : NativeAttrConcept<
                "::mlir::teil::DimSizeArrayAttr",
                "dimension sizes"
        >;

def TeIL_Shape
        : NativeTypeConcept<"::mlir::teil::ShapeType", "shape">;

def TeIL_ShapeType
        : ConstrainedTypeAttr<TeIL_Shape>;

def TeIL_DenseShapeAttr
        : NativeAttrConcept<"::mlir::teil::DenseShapeAttr", "dense shape"> {
    let valueType = TeIL_Shape;
    let constBuilderCall = [{
        ::mlir::teil::DenseShapeAttr::get($_builder.getContext(), $0)
    }];
}

#endif
