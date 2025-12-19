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
class SWE2d : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using PDESystem<FloatingPointType, PDEOptions>::InitPDESys;

  ///
  /// \brief Default constructor.
  /// Creates a default SWE2d instance, with default values for model and grid parameters
  ///
  SWE2d()
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SWE2d, PDEParams<PDEOptions>(2, 3, 0, PDEOptions()))
    , m_modelParams(defaultModelParams)
    , m_gridParams(defaultGridParams)
  {
    static_assert(m_gridParams.type == PDEType::SWE2d);
    InitPDESys();
  }

  ///
  /// \brief Constructor
  /// Creates a SWE2d instance with the given model parameters, number of fields, and grid
  /// parameters
  /// \param[in] modelParams The model parameters
  /// \param[in] nfield The number of fields
  /// \param[in] gridParams The grid parameters
  ///
  SWE2d(
    Model2dParams<FloatingPointType> const& modelParams, int nfield, Grid2dParams const& gridParams)
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SWE2d, PDEParams<PDEOptions>(2, nfield, 0, PDEOptions()))
    , m_modelParams(modelParams)
    , m_gridParams(gridParams)
  {
    static_assert(m_gridParams.type == PDEType::SWE2d);
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

  ///
  /// \brief Get the grid parameters for this PDE system
  /// \returns The grid parameters for the PDE system
  ///
  auto GetGridParams() const noexcept -> Grid2dParams
  {
    return m_gridParams;
  }

private:
  Model2dParams<FloatingPointType> m_modelParams {};
  static inline const Model2dParams<FloatingPointType> defaultModelParams {.g = 9.8,
    .nu = 1.0,
    .gammat = 0.1,
    .gamman = 0.8,
    .nm = 1.0,
    .wsurf = 2.0,
    .radi = 0.7,
    .wsurfydir = 1.5};
  Grid2dParams m_gridParams {};
  static inline const Grid2dParams defaultGridParams {.type = PDEType::SWE2d,
    .sideLengths = {1.0, 0.1},
    .spacings = {0.01, 0.001},
    .totalT = 10,
    .dt = 0.01};
};

template<std::floating_point FloatingPointType, typename PDEOptions>
class SWE3d : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using PDESystem<FloatingPointType, PDEOptions>::InitPDESys;

  ///
  /// \brief Default constructor
  /// Creates a default SWE3d instance, with default values for model and grid parameters
  ///
  SWE3d()
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SWE3d, PDEParams<PDEOptions>(2, 3, 0, PDEOptions()))
    , m_modelParams(defaultModelParams)
    , m_gridParams(defaultGridParams)
  {
    static_assert(m_gridParams.type == PDEType::SWE3d);
    InitPDESys();
  }

  ///
  /// \brief Constructor
  /// \param[in] modelParams The model parameters
  /// \param[in] nfield The number of fields
  /// \param[in] gridParams The grid parameters
  ///
  SWE3d(
    Model3dParams<FloatingPointType> const& modelParams, int nfield, Grid3dParams const& gridParams)
    : PDESystem<FloatingPointType, PDEOptions>(
        PDEType::SWE3d, PDEParams<PDEOptions>(2, nfield, 0, PDEOptions()))
    , m_modelParams(modelParams)
    , m_gridParams(gridParams)
  {
    static_assert(m_gridParams.type == PDEType::SWE3d);
    InitPDESys();
  }

  ///
  /// \brief Get the model parameters for this PDE system
  /// \returns The model parameters for the PDE system
  ///
  auto GetModelParams() const noexcept -> Model3dParams<FloatingPointType>
  {
    return m_modelParams;
  }

  ///
  /// \brief Get the grid parameters for this PDE system
  /// \returns The grid parameters for the PDE system
  ///
  auto GetGridParams() const noexcept -> Grid3dParams
  {
    return m_gridParams;
  }

private:
  Model3dParams<FloatingPointType> m_modelParams {};
  static constexpr Model3dParams<FloatingPointType> defaultModelParams {.g = 9.8,
    .nu = 1.0,
    .gammat = 0.1,
    .gamman = 0.8,
    .nm = 1.0,
    .wsurf = 2.0,
    .radi = 0.7,
    .wsurfydir = 0.0,
    .wsurfzdur = 0.0};
  Grid3dParams m_gridParams {};
  static constexpr Grid3dParams defaultGridParams {.type = PDEType::SWE3d,
    .sideLengths = {0, 0, 0},
    .spacings = {0, 0, 0},
    .totalT = 0.0,
    .dt = 0.0};
};

}  // namespace fcfd::pdemodel
