#pragma once

#include <array>
#include <concepts>

namespace fcfd::pdemodel
{

// Base 1D model parameters for SWE or SVE
template<std::floating_point FloatingPointType>
struct Model1dParams
{
  constexpr static FloatingPointType defaultG = static_cast<FloatingPointType>(9.81);

  FloatingPointType g = defaultG;  // gravity acceleration
  FloatingPointType nu = 0.0;  // viscosity
  FloatingPointType gammat = 0.0;  // tangential friction
  FloatingPointType gamman = 0.0;  // normal friction
  FloatingPointType nm = 0.0;  // Manning coefficient
  FloatingPointType wsurf = 0.0;  // free surface height / width
  FloatingPointType radi = 0.0;  // characteristic radius
};

// 2D model parameters extend 1D
template<std::floating_point FloatingPointType>
struct Model2dParams
{
  constexpr static FloatingPointType defaultG = static_cast<FloatingPointType>(9.81);

  FloatingPointType g = defaultG;  // gravity acceleration
  FloatingPointType nu = 0.0;  // viscosity
  FloatingPointType gammat = 0.0;  // tangential friction
  FloatingPointType gamman = 0.0;  // normal friction
  FloatingPointType nm = 0.0;  // Manning coefficient
  FloatingPointType wsurf = 0.0;  // free surface height / width
  FloatingPointType radi = 0.0;  // characteristic radius
  FloatingPointType wsurfydir = 0.0;  // surface variation in y-direction
  FloatingPointType nuY = 0.0;  // optional: viscosity in y-direction
  FloatingPointType gammatY = 0.0;  // optional: tangential friction in y
};

// Aliases for convenience
using Model1d = Model1dParams<double>;
using Model2d = Model2dParams<double>;

}  // namespace fcfd::pdemodel
