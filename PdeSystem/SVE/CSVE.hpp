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
class CasulliSVE1 : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using PDESystem<FloatingPointType, PDEOptions>::InitPDESys;

  ///
  /// \brief Default constructor
  /// Casulli 1D free-surface hydrodynamics model
  /// Primary unknowns: (eta, u)
  ///
  CasulliSVE1()
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::CasulliSVE1d,
        PDEParams<PDEOptions>(1, 2, 0, PDEOptions()))
    , m_modelParams(defaultModelParams)
    , m_gridParams(defaultGridParams)
  {
    static_assert(m_gridParams.type == PDEType::CasulliSVE1d);
    InitPDESys();
  }

  ///
  /// \brief Constructor
  ///
  CasulliSVE1(
    Model1dParams<FloatingPointType> const& modelParams,
    int nfield,
    Grid1dParams const& gridParams)
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::CasulliSVE1d,
        PDEParams<PDEOptions>(1, nfield, 0, PDEOptions()))
    , m_modelParams(modelParams)
    , m_gridParams(gridParams)
  {
    static_assert(m_gridParams.type == PDEType::CasulliSVE1d);
    InitPDESys();
  }

  ///
  /// \brief Get model parameters
  ///
  auto GetModelParams() const noexcept -> Model1dParams<FloatingPointType>
  {
    return m_modelParams;
  }

  ///
  /// \brief Get grid parameters
  ///
  auto GetGridParams() const noexcept -> Grid1dParams
  {
    return m_gridParams;
  }

private:
  Model1dParams<FloatingPointType> m_modelParams {};

  static constexpr Model1dParams<FloatingPointType> defaultModelParams {
    .g = 9.8,
    .nu = 1.0,
    .gammat = 0.0,
    .gamman = 0.0,
    .nm = 1.0,
    .wsurf = 0.0,
    .radi = 0.0
  };

  Grid1dParams m_gridParams {};

  static constexpr Grid1dParams defaultGridParams {
    .type = PDEType::CasulliSVE1d,
    .sideLengths = {1.0},
    .spacings = {0.01},
    .totalT = 10.0,
    .dt = 0.1
  };
};

} // namespace fcfd::pdemodel
