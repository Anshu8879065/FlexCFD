#pragma once

#include <concepts>

namespace fcfd::pdemodel
{

template<std::semiregular PDEOptions>
struct PDEParams
{
  PDEOptions myopts {};

  //! Number of dimensions
  int nd {};

  //! Number of variables per grid cell
  int nv {};

  //! Number of boundary conditions
  int nbound {};

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  constexpr explicit PDEParams(int numDims, int numGridVars) noexcept
    : nd(numDims)
    , nv(numGridVars)
  {
  }
};

}  // namespace fcfd::pdemodel
