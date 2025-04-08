//===- ContiguousType.td -----------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL ContiguousType interface.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_CONTIGUOUSTYPEINTERFACE
#define EKL_CONTIGUOUSTYPEINTERFACE

include "mlir/IR/OpBase.td"

def EKL_ContiguousType : TypeInterface<"ContiguousType"> {
    let cppNamespace = "::mlir::ekl";

    let description = [{
        Interface for a type that represents an aggregate of a scalars in
        contiguous memory.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Gets the underlying scalar type.
            }],
            /*retTy=*/"::mlir::Type",
            /*methodName=*/"getScalarType",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/[{
                Gets the shape of the aggregate type.
            }],
            /*retTy=*/"::mlir::ekl::ShapeRef",
            /*methodName=*/"getShape",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">];

    code extraSharedClassDeclaration = [{
        /// Determines whether this aggregate is scalar like.
        ///
        /// A scalar-like aggregate has an empty shape.
        auto isScalarLike() const -> bool { return $_type.getShape().empty(); }

        /// Gets the number of extents.
        auto getNumExtents() const -> unsigned
        {
            return static_cast<unsigned>($_type.getShape().size());
        }

        /// Obtains the extent for @p dim .
        ///
        /// @pre    `dim < getNumExtents()`
        auto getExtent(unsigned dim) const -> Extent
        {
            return $_type.getShape()[dim];
        }

        /// Tries to obtain the total number of elements in the aggregate.
        ///
        /// See flatten(ShapeRef) for more information.
        ///
        /// @retval failure Result is out of range.
        /// @retval Extent  Total number of contained elements.
        auto getNumElements() const -> FailureOr<Extent>
        {
            return flatten($_type.getShape());
        }
    }];
}

#endif // EKL_CONTIGUOUSTYPEINTERFACE
