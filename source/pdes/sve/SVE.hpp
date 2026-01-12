#pragma once

#include <cmath>  // std::abs, std::pow
#include <concepts>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "pdes/BoundaryCondition.hpp"
#include "pdes/ModelParams.hpp"
#include "pdes/PDEParams.hpp"
#include "pdes/PDESystem.hpp"
#include "pdes/PDEType.hpp"

namespace fcfd::pdemodel
{

// =========================================================================
// SVE (1D Saint-Venant Equations, conservative form)
// State: U = [A, Q]^T
// Flux:  F(U) = [ Q,  Q^2/A + g A^2/2 ]^T
// Source (momentum): g A (S0 - Sf),   Sf = n_M^2 Q |Q| / A^(10/3),   S0 = -dz_b/dx
// =========================================================================

template<std::floating_point FloatingPointType, typename PDEOptions>
class SVE final : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using Base = PDESystem<FloatingPointType, PDEOptions>;
  using Base::InitPDESys;

  SVE()
    : Base(PDEType::SVE, PDEParams<PDEOptions>(1, 2))
    , m_modelParams(defaultModelParams)
  {
    SetBottom();
    SetInitCond();
    SetBdryCond();

    SetLHSOp();
    SetRHSFunc();
    SetJacLHSOp();
    SetJacRHS();

    InitPDESys();
  }

  explicit SVE(Model1dParams<FloatingPointType> const& modelParams)
    : Base(PDEType::SVE, PDEParams<PDEOptions>(1, 2))
    , m_modelParams(modelParams)
  {
    SetBottom();
    SetInitCond();
    SetBdryCond();

    SetLHSOp();
    SetRHSFunc();
    SetJacLHSOp();
    SetJacRHS();

    InitPDESys();
  }

  auto GetModelParams() const noexcept -> Model1dParams<FloatingPointType>
  {
    return m_modelParams;
  }

  // -----------------------------
  // SVE: Initial condition
  // Returns the initial conservative state
  //   U0(x) = [A0, Q0]
  // -----------------------------
  void SetInitCond() override
  {
    this->Base::SetInitCond(
      [](std::vector<FloatingPointType> const& /*x*/) -> std::vector<FloatingPointType>
      {
        return std::vector<FloatingPointType> {
          FloatingPointType(20.0),  // A0
          FloatingPointType(20.0)   // Q0
        };
      });
  }

  auto InitCond(std::vector<FloatingPointType> const&) const -> std::vector<FloatingPointType> override
  {
    return std::vector<FloatingPointType>({FloatingPointType(20), FloatingPointType(20)});
  }

  // -----------------------------
  // SVE: Boundary conditions
  // IDs:
  //   1 = inlet (x=0)
  //   2 = outlet (x=L)
  // -----------------------------
  void SetBdryCond() override
  {
    std::unordered_map<int, BoundaryCondition<FloatingPointType>> bcs;

    bcs[1] = BoundaryCondition<FloatingPointType>(
      BoundaryType::Dirichlet,
      [](auto const&, auto) -> std::vector<FloatingPointType>
      {
        return std::vector<FloatingPointType> {
          FloatingPointType(20.0),  // A
          FloatingPointType(20.0)   // Q
        };
      });

    bcs[2] = BoundaryCondition<FloatingPointType>(
      BoundaryType::Dirichlet,
      [](auto const&, auto) -> std::vector<FloatingPointType>
      {
        return std::vector<FloatingPointType> {
          FloatingPointType(20.0),  // A
          FloatingPointType(20.0)   // Q
        };
      });

    this->Base::SetBdryCond(std::move(bcs), /*numConds=*/2);
  }

  auto BdryCond() -> FloatingPointType override
  {
    return FloatingPointType {};
  }

  void SetBottom() override
  {
    Base::SetBottom(
      [](std::span<const FloatingPointType> /*coords*/) -> std::vector<FloatingPointType>
      {
        return {FloatingPointType(0)};
      });
  }

  // -----------------------------
  // RHS source term: only momentum source is non-zero
  // -----------------------------
  void SetRHSFunc() override
  {
    Base::SetRHSFunc(
      [this](std::vector<FloatingPointType> const& input, std::vector<FloatingPointType>& out, int /*field*/) -> void
      {
        if (out.size() != 2) {
          out.resize(2);
        }

        const FloatingPointType A = input[0];
        const FloatingPointType Q = input[1];

        const FloatingPointType g = m_modelParams.g;
        const FloatingPointType nM = m_modelParams.nm;

        FloatingPointType Sf = FloatingPointType(0);
        if (A > m_dryTol) {
          Sf = (nM * nM) * Q * std::abs(Q) / std::pow(A, FloatingPointType(10.0 / 3.0));
        }

        out[0] = FloatingPointType(0);
        out[1] = g * A * (m_S0 - Sf);
      });
  }

  // -----------------------------
  // LHS split:
  // A-split: U_t
  // B-split: F(U)_x
  // Signature matches NEW PDESystem::SystemLHSFunctionType
  // -----------------------------
  void SetLHSOp() override
  {
    auto lhsa = [](std::vector<FloatingPointType> const& /*input*/,
                  std::vector<FloatingPointType> const& vdot,
                  std::vector<FloatingPointType> const& /*dvdx*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dvdy*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dvdz*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv2dx*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv2dy*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv2dz*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv3dy*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv3dz*/,
                  std::vector<FloatingPointType>& out) -> void
    {
      if (out.size() != 2) {
        out.resize(2);
      }
      out[0] = vdot[0];
      out[1] = vdot[1];
    };

    auto lhsb = [this](std::vector<FloatingPointType> const& input,
                  std::vector<FloatingPointType> const& /*vdot*/,
                  std::vector<FloatingPointType> const& dvdx,
                  std::optional<std::vector<FloatingPointType>> const& /*dvdy*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dvdz*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv2dx*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv2dy*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv2dz*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv3dy*/,
                  std::optional<std::vector<FloatingPointType>> const& /*dv3dz*/,
                  std::vector<FloatingPointType>& out) -> void
    {
      if (out.size() != 2) {
        out.resize(2);
      }

      const FloatingPointType A = input[0];
      const FloatingPointType Q = input[1];
      const FloatingPointType Ax = dvdx[0];
      const FloatingPointType Qx = dvdx[1];

      const FloatingPointType g = m_modelParams.g;

      out[0] = Qx;

      FloatingPointType flux2_x = FloatingPointType(0);
      if (A > m_dryTol) {
        const FloatingPointType dF2dA = (g * A) - (Q * Q) / (A * A);
        const FloatingPointType dF2dQ = (FloatingPointType(2) * Q) / A;
        flux2_x = dF2dA * Ax + dF2dQ * Qx;
      }

      out[1] = flux2_x;
    };

    Base::SetLHSOp(lhsa, lhsb);
  }

  // -----------------------------
  // Jacobian of LHS (A-split and B-split)
  // Signature matches NEW PDESystem::SystemLHSJacFunctionType
  // -----------------------------
  void SetJacLHSOp() override
  {
    auto jac_lhsa = [](std::vector<FloatingPointType> const& /*input*/,
                      std::vector<FloatingPointType> const& /*vdot*/,
                      std::vector<FloatingPointType> const& /*dvdx*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dvdy*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dvdz*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv2dx*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv2dy*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv2dz*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv3dy*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv3dz*/,
                      std::vector<FloatingPointType>& out,
                      int rowo,
                      int colo,
                      int derivo) -> void
    {
      if (out.size() != 2) {
        out.assign(2, FloatingPointType(0));
      }

      if (derivo == 0) {
        out[0] = (rowo == 0 && colo == 0) ? FloatingPointType(1) : FloatingPointType(0);
        out[1] = (rowo == 1 && colo == 1) ? FloatingPointType(1) : FloatingPointType(0);
      }
    };

    auto jac_lhsb = [this](std::vector<FloatingPointType> const& input,
                      std::vector<FloatingPointType> const& /*vdot*/,
                      std::vector<FloatingPointType> const& /*dvdx*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dvdy*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dvdz*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv2dx*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv2dy*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv2dz*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv3dy*/,
                      std::optional<std::vector<FloatingPointType>> const& /*dv3dz*/,
                      std::vector<FloatingPointType>& out,
                      int rowo,
                      int colo,
                      int derivo) -> void
    {
      if (out.size() != 2) {
        out.assign(2, FloatingPointType(0));
      }
      if (derivo != 1) {
        return;
      }

      const FloatingPointType A = input[0];
      const FloatingPointType Q = input[1];
      const FloatingPointType g = m_modelParams.g;

      FloatingPointType dF2dA = FloatingPointType(0);
      FloatingPointType dF2dQ = FloatingPointType(0);
      if (A > m_dryTol) {
        dF2dA = (g * A) - (Q * Q) / (A * A);
        dF2dQ = (FloatingPointType(2) * Q) / A;
      }

      if (rowo == 0 && colo == 0) {
        out[0] = FloatingPointType(0);
      }
      if (rowo == 0 && colo == 1) {
        out[0] = FloatingPointType(1);
      }

      if (rowo == 1 && colo == 0) {
        out[1] = dF2dA;
      }
      if (rowo == 1 && colo == 1) {
        out[1] = dF2dQ;
      }
    };

    Base::SetJacLHSOp(jac_lhsa, jac_lhsb);
  }

  // -----------------------------
  // Jacobian of RHS source S(U)
  // out[0] = dS2/dA, out[1] = dS2/dQ
  // -----------------------------
  void SetJacRHS() override
  {
    Base::SetJacRHS(
      [this](std::vector<FloatingPointType> const& input, std::vector<FloatingPointType>& out, int /*field*/) -> void
      {
        if (out.size() != 2) {
          out.assign(2, FloatingPointType(0));
        }

        const FloatingPointType A = input[0];
        const FloatingPointType Q = input[1];

        const FloatingPointType g = m_modelParams.g;
        const FloatingPointType nM = m_modelParams.nm;

        if (A <= m_dryTol) {
          return;
        }

        const FloatingPointType absQ = std::abs(Q);
        const FloatingPointType A_pow_10_3 = std::pow(A, FloatingPointType(10.0 / 3.0));
        const FloatingPointType Sf = (nM * nM) * Q * absQ / A_pow_10_3;

        const FloatingPointType dSf_dQ = (nM * nM) * (FloatingPointType(2) * absQ) / A_pow_10_3;

        const FloatingPointType dSf_dA =
          -(FloatingPointType(10.0 / 3.0)) * (nM * nM) * Q * absQ / std::pow(A, FloatingPointType(13.0 / 3.0));

        out[0] = g * (m_S0 - Sf) + g * A * (-dSf_dA);
        out[1] = g * A * (-dSf_dQ);
      });
  }

private:
  Model1dParams<FloatingPointType> m_modelParams {};

  static constexpr Model1dParams<FloatingPointType> defaultModelParams {.g = FloatingPointType(9.8),
    .nu = FloatingPointType(1.0),
    .gammat = FloatingPointType(0.1),
    .gamman = FloatingPointType(0.8),
    .nm = FloatingPointType(0.03),
    .wsurf = FloatingPointType(2.0),
    .radi = FloatingPointType(0.7)};

  FloatingPointType m_S0 {FloatingPointType(0)};
  FloatingPointType m_dryTol {FloatingPointType(1e-12)};
};

}  // namespace fcfd::pdemodel
