#pragma once

#include <concepts>
#include <array>

namespace fcfd::pdemodel
{

// ---------------------------
// Base 1D model parameters for SWE/SVE
// ---------------------------
template<std::floating_point FloatingPointType>
struct ModelParams
{
    constexpr static FloatingPointType defaultG = static_cast<FloatingPointType>(9.81);

    FloatingPointType g       = defaultG;  // gravity acceleration
    FloatingPointType nu      = 0.0;      // viscosity
    FloatingPointType gammat  = 0.0;      // tangential friction
    FloatingPointType gamman  = 0.0;      // normal friction
    FloatingPointType nm      = 0.0;      // Manning coefficient
    FloatingPointType wsurf   = 0.0;      // free surface height / width
    FloatingPointType radi    = 0.0;      // characteristic radius
};

// ---------------------------
// 2D model parameters extend 1D
// ---------------------------
template<std::floating_point FloatingPointType>
struct Model2dParams : ModelParams<FloatingPointType>
{
    FloatingPointType wsurfY   = 0.0;  // surface variation in y-direction
    FloatingPointType nuY      = 0.0;  // optional: viscosity in y-direction
    FloatingPointType gammatY  = 0.0;  // optional: tangential friction in y
};

// ---------------------------
// 3D model parameters extend 2D
// ---------------------------
template<std::floating_point FloatingPointType>
struct Model3dParams : Model2dParams<FloatingPointType>
{
    FloatingPointType wsurfZ   = 0.0;  // surface variation in z-direction
    FloatingPointType nuZ      = 0.0;  // optional: viscosity in z-direction
    FloatingPointType gammatZ  = 0.0;  // optional: tangential friction in z
};

// ---------------------------
// Casulli 2D model parameters
// ---------------------------
template<std::floating_point FloatingPointType>
struct Casulli2dParams : Model2dParams<FloatingPointType>
{
    // Casulli-specific parameters
    FloatingPointType Gamma   = 0.0;   // Casulli scheme parameter
    FloatingPointType GammaT  = 0.0;   // tangential forcing coefficient
    FloatingPointType ua      = 0.0;   // external / atmospheric velocity
};

// ---------------------------
// Aliases for convenience
// ---------------------------
using Model1d      = ModelParams<double>;
using Model2d      = Model2dParams<double>;
using Model3d      = Model3dParams<double>;
using Casulli2d    = Casulli2dParams<double>;

} // namespace fcfd::pdemodel

