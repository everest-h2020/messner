/// Declares the EVP conversion passes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Conversion/EKLToFunc/EKLToFunc.h"
#include "messner/Conversion/EKLToLinalg/EKLToLinalg.h"
#include "messner/Conversion/EKLToStandard/EKLToStandard.h"

namespace messner {

#define GEN_PASS_REGISTRATION
#include "messner/Conversion/Passes.h.inc"

} // namespace messner
