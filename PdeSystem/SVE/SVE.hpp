#pragma once

#include <concepts>

#include "ModelParams.hpp"
#include "PDEParams.hpp"
#include "PDESystem.hpp"
#include "PDEType.hpp"

namespace fcfd::pdemodel
{

template<std::floating_point FloatingPointType, typename PDEOptions>
class SVE : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using PDESystem<FloatingPointType, PDEOptions>::InitPDESys;

  ///
  /// \brief Default constructor.
  /// Creates a default SVE instance, with default values for model parameters
  ///
  SVE()
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SVE, PDEParams<PDEOptions>(1, 2, 0, PDEOptions()))
    , m_modelParams(defaultModelParams)
  {
    InitPDESys();
  }

  ///
  /// \brief Constructor
  /// Creates a SVE instance with the given model parameters and number of fields
  /// \param[in] modelParams The model parameters
  /// \param[in] nfield The number of fields
  ///
  SVE(Model1dParams<FloatingPointType> const& modelParams, int nfield)
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SVE, PDEParams<PDEOptions>(1, nfield, 0, PDEOptions()))
    , m_modelParams(modelParams)
  {
    InitPDESys();
  }

  ///
  /// \brief Get the model parameters for this PDE system
  /// \returns The model parameters for the PDE system
  ///
  auto GetModelParams() -> Model1dParams<FloatingPointType>
  {
    return m_modelParams;
  }

private:
  Model1dParams<FloatingPointType> m_modelParams {};
  
  static constexpr Model1dParams<FloatingPointType> defaultModelParams {
    .g = 9.8, .nu = 1.0, .gammat = 0.1, .gamman = 0.8, .nm = 1.0, .wsurf = 2.0, .radi = 0.7
  };
};

} // namespace fcfd::pdemodel
