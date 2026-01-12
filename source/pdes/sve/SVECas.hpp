// SVECas.hpp
#pragma once

#include <algorithm>
#include <cmath>
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

// 1D subcritical BCs you want: only eta or u (NO Q)
enum class SubcriticalBC : int
{
  StageEta,   // prescribe eta
  VelocityU   // prescribe u
};

struct SubcriticalBCSpec
{
  SubcriticalBC left = SubcriticalBC::StageEta;
  SubcriticalBC right = SubcriticalBC::StageEta;
};

template<std::floating_point Real>
struct CasulliBC1D
{
  SubcriticalBCSpec spec {};

  // meaning depends on spec:
  // StageEta  -> value is eta
  // VelocityU -> value is u
  Real left_value = Real(0);
  Real right_value = Real(0);

  // Only used when StageEta is prescribed and you also want to force a u on that boundary
  Real left_u_if_eta = Real(0);
  Real right_u_if_eta = Real(0);
};

template<std::floating_point FloatingPointType, std::semiregular PDEOptions>
class SVEcas final : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using Base = PDESystem<FloatingPointType, PDEOptions>;
  using Real = FloatingPointType;
  using ModelParamT = fcfd::pdemodel::Model1dParams<Real>;

  using Base::InitPDESys;

  SVEcas()
    : Base(PDEType::SVEcas, PDEParams<PDEOptions>(1, 2))
  {
    InstallNoOpHooks();
    InitPDESys();
  }

  explicit SVEcas(ModelParamT const& mp)
    : Base(PDEType::SVEcas, PDEParams<PDEOptions>(1, 2))
    , m_modelParams(mp)
  {
    InstallNoOpHooks();
    InitPDESys();
  }

  // -----------------------
  // model params access (keep ALL these for compatibility)
  // -----------------------
  auto ModelParamsRef() noexcept -> ModelParamT&
  {
    return m_modelParams;
  }

  auto ModelParamsRef() const noexcept -> ModelParamT const&
  {
    return m_modelParams;
  }

  auto GetModelParams() noexcept -> ModelParamT&
  {
    return m_modelParams;
  }

  auto GetModelParams() const noexcept -> ModelParamT const&
  {
    return m_modelParams;
  }

  auto ModelParams() noexcept -> ModelParamT&
  {
    return m_modelParams;
  }

  auto ModelParams() const noexcept -> ModelParamT const&
  {
    return m_modelParams;
  }

  // -----------------------
  // user setters
  // -----------------------
  void SetIC(Real eta0, Real u0)
  {
    m_eta0 = eta0;
    m_u0 = u0;
  }

  // Bottom: constant depth h0; optional slope S0 (so h(x)=h0-S0*x)
  void SetBed(Real h0, Real S0 = Real(0))
  {
    m_h0 = h0;
    m_S0 = S0;
  }

  void SetWindSpeedU(Real ua)
  {
    m_ua = ua;
  }

  void SetBCSpec(SubcriticalBCSpec spec)
  {
    m_bc.spec = spec;
  }

  // stage BC
  void SetLeftEta(Real eta, Real u_if_eta = Real(0))
  {
    m_bc.spec.left = SubcriticalBC::StageEta;
    m_bc.left_value = eta;
    m_bc.left_u_if_eta = u_if_eta;
  }

  void SetRightEta(Real eta, Real u_if_eta = Real(0))
  {
    m_bc.spec.right = SubcriticalBC::StageEta;
    m_bc.right_value = eta;
    m_bc.right_u_if_eta = u_if_eta;
  }

  // velocity BC
  void SetLeftU(Real u)
  {
    m_bc.spec.left = SubcriticalBC::VelocityU;
    m_bc.left_value = u;
  }

  void SetRightU(Real u)
  {
    m_bc.spec.right = SubcriticalBC::VelocityU;
    m_bc.right_value = u;
  }

  // -----------------------
  // CasulliWrap getters
  // -----------------------
  auto Eta0() const noexcept -> Real { return m_eta0; }
  auto U0() const noexcept -> Real { return m_u0; }

  auto BC() const noexcept -> CasulliBC1D<Real> const& { return m_bc; }

  auto BottomAt(Real x) const noexcept -> Real { return m_h0 - m_S0 * x; }

  static auto HfromEta(Real h, Real eta) noexcept -> Real
  {
    return std::max(Real(0), h + eta);
  }

  auto WindSpeedU() const noexcept -> Real { return m_ua; }
  auto DryTol() const noexcept -> Real { return m_dryTol; }

  // Manning + gamma_T
  auto GammaFace(Real H_face, Real u_star) const noexcept -> Real
  {
    const Real g = m_modelParams.g;
    const Real n = m_modelParams.nm;
    const Real gammaT = m_modelParams.gammat;

    const Real R = (m_modelParams.radi > m_dryTol) ? m_modelParams.radi : std::max(H_face, m_dryTol);
    const Real gammaB = g * (n * n) * std::abs(u_star) / std::cbrt(R);
    return gammaB + gammaT;
  }

  // -----------------------
  // PDESystem virtual hooks (kept as no-ops / placeholders)
  // -----------------------
  void SetBottom() override {}
  void SetInitCond() override {}

  auto InitCond(std::vector<Real> const&) const -> std::vector<Real> override
  {
    return std::vector<Real>({Real(0), Real(0)});
  }

  void SetBdryCond() override {}
  auto BdryCond() -> Real override { return Real {}; }

  void SetRHSFunc() override {}
  void SetJacRHS() override {}
  void SetLHSOp() override {}
  void SetJacLHSOp() override {}

private:
  void InstallNoOpHooks()
  {
    // bottom uses coords span
    Base::SetBottom(
      [this](std::span<const Real> coords) -> std::vector<Real>
      {
        const Real x = coords.empty() ? Real(0) : coords[0];
        return {BottomAt(x)};
      });

    // init cond uses vector coords (ignored here)
    Base::SetInitCond(
      [this](std::vector<Real> const&) -> std::vector<Real>
      {
        const Real h0 = BottomAt(Real(0));
        const Real H0 = HfromEta(h0, m_eta0);
        return {H0, m_u0};
      });

    // Placeholder BCs for framework completeness (CasulliWrap uses BC() above)
    std::unordered_map<int, BoundaryCondition<Real>> bcs;
    auto transmissive = [](std::vector<Real> const& u_inner, Real) -> std::vector<Real> { return u_inner; };
    bcs[1] = BoundaryCondition<Real>(BoundaryType::Neumann, transmissive);
    bcs[2] = BoundaryCondition<Real>(BoundaryType::Neumann, transmissive);
    Base::SetBdryCond(std::move(bcs), 2);

    // No-op RHS/JacRHS
    Base::SetRHSFunc([](std::vector<Real> const&, std::vector<Real>& out, int) -> void { out.assign(2, Real(0)); });
    Base::SetJacRHS([](std::vector<Real> const&, std::vector<Real>& out, int) -> void { out.assign(2, Real(0)); });

    // LHS A: pass-through vdot
    auto lhsa = [](std::vector<Real> const&,
                  std::vector<Real> const& vdot,
                  std::vector<Real> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::vector<Real>& out) -> void
    {
      out = vdot;
      if (out.size() != 2) {
        out.resize(2, Real(0));
      }
    };

    // LHS B: zero
    auto lhsb = [](std::vector<Real> const&,
                  std::vector<Real> const&,
                  std::vector<Real> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::vector<Real>& out) -> void
    {
      out.assign(2, Real(0));
    };

    Base::SetLHSOp(lhsa, lhsb);

    // JacLHS no-op
    auto jac0 = [](std::vector<Real> const&,
                  std::vector<Real> const&,
                  std::vector<Real> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::optional<std::vector<Real>> const&,
                  std::vector<Real>& out,
                  int,
                  int,
                  int) -> void
    {
      out.assign(2, Real(0));
    };

    Base::SetJacLHSOp(jac0, jac0);
  }

private:
  ModelParamT m_modelParams {};

  Real m_eta0 {Real(0)};
  Real m_u0 {Real(0)};

  Real m_h0 {Real(0)};
  Real m_S0 {Real(0)};

  Real m_ua {Real(0)};

  CasulliBC1D<Real> m_bc {};
  Real m_dryTol {Real(1e-12)};
};

}  // namespace fcfd::pdemodel
