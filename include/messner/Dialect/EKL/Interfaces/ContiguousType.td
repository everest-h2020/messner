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
        Interface for a type that represents an aggregate of a scalar type in
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
                Gets the extents of the aggregate type.
            }],
            /*retTy=*/"::mlir::ekl::ExtentRange",
            /*methodName=*/"getExtents",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">];

    code extraSharedClassDeclaration = [{
        /// Determines whether this aggregate is a scalar value.
        [[nodiscard]] bool isScalar() const
        {
            return ::mlir::ekl::isScalar($_type.getExtents());
        }

        /// Gets the number of extents.
        [[nodiscard]] std::size_t getNumExtents() const
        {
            return $_type.getExtents().size();
        }
        /// Obtains the extent for @p dim .
        ///
        /// @pre    `dim < getNumExtents()`
        [[nodiscard]] ::mlir::ekl::extent_t getExtent(unsigned dim) const
        {
            return $_type.getExtents()[dim];
        }

        /// Tries to obtain the total number of elements in the aggregate.
        ///
        /// See flatten(ExtentRange) for more information.
        ///
        /// @retval failure     Result is out of range of extent_t.
        /// @retval extent_t    Number of contained elements.
        [[nodiscard]] ::mlir::FailureOr<::mlir::ekl::extent_t> getNumElements()
        {
            return ::mlir::ekl::flatten($_type.getExtents());
        }
    }];
}

#endif
