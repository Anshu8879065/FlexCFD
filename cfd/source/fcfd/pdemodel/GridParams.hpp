#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "PDEType.hpp"

namespace fcfd::pdemodel
{

template<std::size_t Dim>
struct GridParams
{
  PDEType type;
  // std::array<std::size_t, Dim> numPoints;
  std::array<double, Dim> sideLengths;
  std::array<double, Dim> spacings;
  double totalT;
  double dt;

  // automatic spacing calculation
  // void compute_spacings() {
  // for (std::size_t i = 0; i < Dim; ++i)
  // spacings[i] = sideLengths[i] / (numPoints[i] - 1);
  //}
};

// Aliases for clarity
using Grid1dParams = GridParams<1>;
using Grid2dParams = GridParams<2>;
using Grid3dParams = GridParams<3>;

}  // namespace fcfd::pdemodel
