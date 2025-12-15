#pragma once

#include <concepts>

namespace fcfd::pdemodel
{

// Base model parameters (1D)
template<std::floating_point FloatingPointType>
struct Model1dParams
{
  constexpr static FloatingPointType defaultG = static_cast<FloatingPointType>(9.81);

  FloatingPointType g = defaultG;  // gravity
  FloatingPointType nu = 0.0;  // viscosity
  FloatingPointType gammat = 0.0;  // tangential friction
  FloatingPointType gamman = 0.0;  // normal friction
  FloatingPointType nm = 0.0;  // Manning coefficient
  FloatingPointType wsurf = 0.0;  // free surface width / height
  FloatingPointType radi = 0.0;  // characteristic radius or other parameters
};

// 2D model parameters extend 1D
template<std::floating_point FloatingPointType>
struct Model2dParams : Model1dParams<FloatingPointType>
{
  FloatingPointType wsurfydir = 0.0;  // e.g., water surface height in y-direction
};

// 3D model parameters extend 2D
template<std::floating_point FloatingPointType>
struct Model3dParams : Model2dParams<FloatingPointType>
{
  FloatingPointType wsurfzdir = 0.0;  // water surface height in z-direction
};

}  // namespace fcfd::pdemodel
