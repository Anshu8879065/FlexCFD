#pragma once

#include <concepts>

#include "ModelParams.hpp"
#include "PDEParams.hpp"
#include "PDESystem.hpp"
#include "PDEType.hpp"

namespace fcfd::pdemodel
{

template<std::floating_point FloatingPointType, typename PDEOptions>
class SWE2d : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using PDESystem<FloatingPointType, PDEOptions>::InitPDESys;

  ///
  /// \brief Default constructor.
  /// Creates a default SWE2d instance, with default values for model parameters
  ///
  SWE2d()
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SWE2d, PDEParams<PDEOptions>(2, 3, 0, PDEOptions()))
    , m_modelParams(defaultModelParams)
  {
    InitPDESys();
  }

  ///
  /// \brief Constructor
  /// Creates a SWE2d instance with the given model parameters and number of fields
  /// \param[in] modelParams The model parameters
  /// \param[in] nfield The number of fields
  ///
  SWE2d(Model2dParams<FloatingPointType> const& modelParams, int nfield)
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SWE2d, PDEParams<PDEOptions>(2, nfield, 0, PDEOptions()))
    , m_modelParams(modelParams)
  {
    InitPDESys();
  }

  ///
  /// \brief Get the model parameters for this PDE system
  /// \returns The model parameters for the PDE system
  ///
  auto GetModelParams() const noexcept -> Model2dParams<FloatingPointType>
  {
    return m_modelParams;
  }

private:
  Model2dParams<FloatingPointType> m_modelParams {};
  
  static constexpr Model2dParams<FloatingPointType> defaultModelParams {
    .g = 9.8,
    .nu = 1.0,
    .gammat = 0.1,
    .gamman = 0.8,
    .nm = 1.0,
    .wsurf = 2.0,
    .radi = 0.7,
    .wsurfydir = 1.5
  };
};

} // namespace fcfd::pdemodel
