// SWE2dCas.hpp
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

enum class SubcriticalBC2D : int
{
  StageEta,  // prescribe eta
  Velocity   // prescribe normal velocity (u on left/right, v on bottom/top)
};

template<std::floating_point Real>
struct CasulliBC2D
{
  SubcriticalBC2D left = SubcriticalBC2D::Velocity;
  SubcriticalBC2D right = SubcriticalBC2D::StageEta;
  SubcriticalBC2D bottom = SubcriticalBC2D::Velocity;
  SubcriticalBC2D top = SubcriticalBC2D::StageEta;

  Real left_value {Real(0)};
  Real right_value {Real(0)};
  Real bottom_value {Real(0)};
  Real top_value {Real(0)};
};

template<std::floating_point FloatingPointType, std::semiregular PDEOptions>
class SWE2dcas final : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using Base = PDESystem<FloatingPointType, PDEOptions>;
  using Real = FloatingPointType;

  using Base::InitPDESys;

  SWE2dcas()
    : Base(PDEType::SWE2dcas, PDEParams<PDEOptions>(2, 3))
  {
    InstallNoOpHooks();
    InitPDESys();
  }

  explicit SWE2dcas(fcfd::pdemodel::Model2dParams<Real> const& mp)
    : Base(PDEType::SWE2dcas, PDEParams<PDEOptions>(2, 3))
    , m_modelParams(mp)
  {
    InstallNoOpHooks();
    InitPDESys();
  }

  auto ModelParams() noexcept -> Model2dParams<Real>& { return m_modelParams; }
  auto ModelParams() const noexcept -> Model2dParams<Real> const& { return m_modelParams; }

  // IC seeds
  void SetIC(Real eta0, Real u0, Real v0)
  {
    m_eta0 = eta0;
    m_u0 = u0;
    m_v0 = v0;
  }

  // constant bottom; optional planar slopes
  void SetBed(Real h0, Real Sx = Real(0), Real Sy = Real(0))
  {
    m_h0 = h0;
    m_Sx = Sx;
    m_Sy = Sy;
  }

  void SetWind(Real ua, Real va)
  {
    m_ua = ua;
    m_va = va;
  }

  void SetBC(CasulliBC2D<Real> bc)
  {
    m_bc = bc;
  }

  // getters for wrap
  auto Eta0() const noexcept -> Real { return m_eta0; }
  auto U0() const noexcept -> Real { return m_u0; }
  auto V0() const noexcept -> Real { return m_v0; }

  auto BC() const noexcept -> CasulliBC2D<Real> const& { return m_bc; }

  auto BottomAt(Real x, Real y) const noexcept -> Real { return m_h0 - m_Sx * x - m_Sy * y; }

  static auto HfromEta(Real h, Real eta) noexcept -> Real
  {
    return std::max(Real(0), h + eta);
  }

  auto WindU() const noexcept -> Real { return m_ua; }
  auto WindV() const noexcept -> Real { return m_va; }
  auto DryTol() const noexcept -> Real { return m_dryTol; }

  auto GammaFace(Real H_face, Real speed_star) const noexcept -> Real
  {
    const Real g = m_modelParams.g;
    const Real n = m_modelParams.nm;
    const Real gammaT = m_modelParams.gammat;

    const Real R = (m_modelParams.radi > m_dryTol) ? m_modelParams.radi : std::max(H_face, m_dryTol);
    const Real gammaB = g * (n * n) * std::abs(speed_star) / std::cbrt(R);

    return gammaB + gammaT;
  }

  // PDESystem hooks as no-ops
  void SetBottom() override {}
  void SetInitCond() override {}

  auto InitCond(std::vector<Real> const&) const -> std::vector<Real> override
  {
    return std::vector<Real>({Real(0), Real(0), Real(0)});
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
    Base::SetBottom(
      [this](std::span<const Real> coords) -> std::vector<Real>
      {
        const Real x = coords.size() > 0 ? coords[0] : Real(0);
        const Real y = coords.size() > 1 ? coords[1] : Real(0);
        return {BottomAt(x, y)};
      });

    Base::SetInitCond(
      [this](std::vector<Real> const&) -> std::vector<Real>
      {
        const Real h0 = BottomAt(Real(0), Real(0));
        const Real H0 = HfromEta(h0, m_eta0);
        return {H0, m_u0, m_v0};
      });

    // transmissive placeholder BC map (CasulliWrap uses BC() instead)
    std::unordered_map<int, BoundaryCondition<Real>> bcs;
    auto transmissive = [](std::vector<Real> const& u_inner, Real) -> std::vector<Real> { return u_inner; };
    bcs[1] = BoundaryCondition<Real>(BoundaryType::Neumann, transmissive);
    bcs[2] = BoundaryCondition<Real>(BoundaryType::Neumann, transmissive);
    bcs[3] = BoundaryCondition<Real>(BoundaryType::Neumann, transmissive);
    bcs[4] = BoundaryCondition<Real>(BoundaryType::Neumann, transmissive);
    Base::SetBdryCond(std::move(bcs), 4);

    // no-op RHS/JacRHS
    Base::SetRHSFunc([](std::vector<Real> const&, std::vector<Real>& out, int) -> void { out.assign(3, Real(0)); });
    Base::SetJacRHS([](std::vector<Real> const&, std::vector<Real>& out, int) -> void { out.assign(3, Real(0)); });

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
      if (out.size() != 3) {
        out.resize(3, Real(0));
      }
    };

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
      out.assign(3, Real(0));
    };

    Base::SetLHSOp(lhsa, lhsb);

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
      out.assign(3, Real(0));
    };

    Base::SetJacLHSOp(jac0, jac0);
  }

private:
  Model2dParams<Real> m_modelParams {};

  Real m_eta0 {Real(0)};
  Real m_u0 {Real(0)};
  Real m_v0 {Real(0)};

  Real m_h0 {Real(0)};
  Real m_Sx {Real(0)};
  Real m_Sy {Real(0)};

  Real m_ua {Real(0)};
  Real m_va {Real(0)};

  CasulliBC2D<Real> m_bc {};
  Real m_dryTol {Real(1e-12)};
};

}  // namespace fcfd::pdemodel
