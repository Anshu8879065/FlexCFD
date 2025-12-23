#pragma once

namespace fcfd::pdemodel
{

enum class PDEType
{
  Unknown,
  SVE,
  SWE2d,
  SWE3d,
  CasulliSVE,
  CasulliSVE2d
};

}  // namespace fcfd::pdemodel
