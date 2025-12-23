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
class CasulliSVE2d : public PDESystem<FloatingPointType, PDEOptions>
{
public:
    using PDESystem<FloatingPointType, PDEOptions>::InitPDESys;

    ///
    /// \brief Default constructor.
    /// Creates a default Casulli 2D instance, with default model and grid parameters
    ///
    CasulliSVE2d()
        : PDESystem<FloatingPointType, PDEOptions>(
            PDEType::CasulliSVE2d, PDEParams<PDEOptions>(2, 3, 0, PDEOptions()))
        , m_modelParams(defaultModelParams)
        , m_gridParams(defaultGridParams)
    {
        static_assert(m_gridParams.type == PDEType::CasulliSVE2d);
        InitPDESys();
    }

    ///
    /// \brief Constructor
    /// Creates a CasulliSVE2d instance with given model parameters, number of fields, and grid
    /// parameters
    /// \param[in] modelParams The model parameters
    /// \param[in] nfield The number of fields
    /// \param[in] gridParams The grid parameters
    ///
    CasulliSVE2d(
        Casulli2dParams<FloatingPointType> const& modelParams, 
        int nfield, 
        Grid2dParams const& gridParams)
        : PDESystem<FloatingPointType, PDEOptions>(
            PDEType::CasulliSVE2d, PDEParams<PDEOptions>(2, nfield, 0, PDEOptions()))
        , m_modelParams(modelParams)
        , m_gridParams(gridParams)
    {
        static_assert(m_gridParams.type == PDEType::CasulliSVE2d);
        InitPDESys();
    }

    ///
    /// \brief Get the model parameters for this PDE system
    /// \returns The Casulli model parameters
    ///
    auto GetModelParams() const noexcept -> Casulli2dParams<FloatingPointType>
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
    Casulli2dParams<FloatingPointType> m_modelParams {};
    static inline const Casulli2dParams<FloatingPointType> defaultModelParams {
        .g      = 9.81,
        .nu     = 1.0,
        .gammat = 0.1,
        .gamman = 0.8,
        .nm     = 1.0,
        .wsurf  = 2.0,
        .radi   = 0.7,
        .wsurfY = 1.5,
        .Gamma  = 0.5,
        .GammaT = 0.2,
        .ua     = 0.0
    };
    Grid2dParams m_gridParams {};
    static inline const Grid2dParams defaultGridParams {
        .type        = PDEType::CasulliSVE2d,
        .sideLengths = {1.0, 0.1},
        .spacings    = {0.01, 0.001},
        .totalT      = 10,
        .dt          = 0.01
    };
};

} // namespace fcfd::pdemodel
