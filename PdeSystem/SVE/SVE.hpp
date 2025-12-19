#pragma once

#include <concepts>

#include "GridParams.hpp"
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
  /// Creates a default SVE instance, with default values for model and grid parameters
  ///
  SVE()
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SVE, PDEParams<PDEOptions>(1, 2, 0, PDEOptions()))
    , m_modelParams(defaultModelParams)
    , m_gridParams(defaultGridParams)
  {
    static_assert(m_gridParams.type == PDEType::SVE);
    InitPDESys();
  }

  ///
  /// \brief Constructor
  /// Creates a SVE instance with the given model parameters, number of fields, and grid
  /// parameters
  /// \param[in] modelParams The model parameters
  /// \param[in] nfield The number of fields
  /// \param[in] gridParams The grid parameters
  ///
  SVE(
    Model1dParams<FloatingPointType> const& modelParams, int nfield, Grid1dParams const& gridParams)
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SVE, PDEParams<PDEOptions>(1, nfield, 0, PDEOptions()))
    , m_modelParams(modelParams)
    , m_gridParams(gridParams)
  {
    static_assert(m_gridParams.type == PDEType::SVE);
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

  ///
  /// \brief Get the grid parameters for this PDE system
  /// \returns The grid parameters for the PDE system
  ///
  auto GetGridParams() -> Grid1dParams
  {
    return m_gridParams;
  }

private:
  Model1dParams<FloatingPointType> m_modelParams {};
  static constexpr Model1dParams<FloatingPointType> defaultModelParams {
    .g = 9.8, .nu = 1.0, .gammat = 0.1, .gamman = 0.8, .nm = 1.0, .wsurf = 2.0, .radi = 0.7};
  Grid1dParams m_gridParams {};
  static constexpr Grid1dParams defaultGridParams {
    .type = PDEType::SVE, .sideLengths = {1.0}, .spacings = {0.01}, .totalT = 10, .dt = 0.01};
};


};

}  // namespace fcfd::pdemodel
