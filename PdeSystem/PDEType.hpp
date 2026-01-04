// PDEType.hpp
#pragma once

#include <cstdint>

namespace fcfd::pdemodel
{

enum class PDEType : uint8_t
{
  Unknown = 0,
  SVE,
  SVEcas,
  SWE2d,
  SWE2dcas,
  SWE3d
};

}  // namespace fcfd::pdemodel
