// Casulliwrap.hpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "CasulliMethodProps.hpp"
#include "numerics/SolTools.hpp"
#include "pdes/PDESystem.hpp"
#include "pdes/PDEType.hpp"

// PDE models
#include "pdes/sve/SVECas.hpp"
#include "pdes/swe/SWE2dCas.hpp"

namespace fcfd::pdenumerics
{

#if !defined(CASE_1D) && !defined(CASE_2D)
#  define CASE_1D
#  define CASE_2D
#endif

struct CasulliAlgOpts
{
};

using CasulliError = int;

template<std::floating_point Real, std::semiregular PDEOptions>
class CasulliWrap final : public SolTool<Real, PDEOptions, CasulliError, CasulliMethodProps<Real>, CasulliAlgOpts>
{
public:
  using BaseTool = SolTool<Real, PDEOptions, CasulliError, CasulliMethodProps<Real>, CasulliAlgOpts>;
  using PDEBase = fcfd::pdemodel::PDESystem<Real, PDEOptions>;

  using BaseTool::InitSolMethod;
  using BaseTool::SetPDE;
  using BaseTool::SetSolOpts;

  CasulliWrap() = default;
  ~CasulliWrap() override = default;

  // ---- safe accessors for main.cpp (avoid protected access) ----
  auto MethodProps() noexcept -> CasulliMethodProps<Real>&
  {
    return this->m_methodProps;
  }

  auto MethodProps() const noexcept -> CasulliMethodProps<Real> const&
  {
    return this->m_methodProps;
  }

  // ---- Newton logging control ----
  void SetNewtonVerbose(bool on) noexcept
  {
    m_newtonVerbose = on;
  }

  // ---- Optional non-uniform initial conditions ----
  // 1D: eta at cell centers, u at faces
  void SetInitialEta1D(std::function<Real(std::size_t i, Real x)> f)
  {
    m_eta_ic_1d = std::move(f);
  }

  void SetInitialUFace1D(std::function<Real(std::size_t f, Real xf)> g)
  {
    m_u_ic_1d = std::move(g);
  }

#if defined(CASE_2D)
  // 2D: eta at cell centers (flattened i+nx*j)
  void SetInitialEta2D(std::function<Real(std::size_t i, std::size_t j, Real x, Real y)> f)
  {
    m_eta_ic_2d = std::move(f);
  }
#endif

  auto InitSolMethod() -> CasulliError override
  {
    return 0;
  }

  auto InitProblem() -> CasulliError
  {
    if (!this->m_myPDE) {
      return 1;
    }
    return 0;
  }

  auto SetSolOpts() -> CasulliError override
  {
    return 0;
  }

  auto NumericalSolve() -> CasulliError override
  {
    if (!this->m_myPDE) {
      return 2;
    }

#if defined(CASE_1D)
    if (auto* p = dynamic_cast<fcfd::pdemodel::SVEcas<Real, PDEOptions>*>(this->m_myPDE.get())) {
      Solve1D_(*p);
      return 0;
    }
#endif

#if defined(CASE_2D)
    if (auto* p = dynamic_cast<fcfd::pdemodel::SWE2dcas<Real, PDEOptions>*>(this->m_myPDE.get())) {
      Solve2D_(*p);
      return 0;
    }
#endif

    return 5;
  }

  auto EvaluateSolution(std::function<std::vector<double>(std::vector<Real>&)> const& interpol) -> CasulliError override
  {
    (void)interpol;  // unused for now
    return CasulliError {};  // or CasulliError::Ok if have an Ok value
  }

  // ------------------------------------------------------------
  // Public access to computed fields
  // ------------------------------------------------------------
  auto Eta1D() const -> std::vector<Real> const&
  {
    return m_eta1d;
  }

  auto UFace1D() const -> std::vector<Real> const&
  {
    return m_uface1d;
  }

  auto Eta2D() const -> std::vector<Real> const&
  {
    return m_eta2d;
  }

  auto UFace2D() const -> std::vector<Real> const&
  {
    return m_uface2d;
  }

  auto VFace2D() const -> std::vector<Real> const&
  {
    return m_vface2d;
  }

  void SetGrid1D(std::size_t nx)
  {
    m_nx = nx;
  }

  void SetGrid2D(std::size_t nx, std::size_t ny)
  {
    m_nx = nx;
    m_ny = ny;
  }

private:
  static auto Idx_(std::size_t i) -> std::size_t
  {
    return i;
  }

  static auto Idx_(std::size_t i, std::size_t j, std::size_t nx) -> std::size_t
  {
    return i + nx * j;
  }

  static auto NormInf_(std::vector<Real> const& v) -> Real
  {
    Real m = Real(0);
    for (auto const& x : v) {
      m = std::max(m, std::abs(x));
    }
    return m;
  }

  static void ThomasSolve_(
    std::vector<Real> a, std::vector<Real> b, std::vector<Real> c, std::vector<Real> const& rhs, std::vector<Real>& x)
  {
    const std::size_t n = b.size();
    x.assign(n, Real(0));
    if (n == 0) {
      return;
    }

    assert(a.size() == (n > 0 ? n - 1 : 0));
    assert(c.size() == (n > 0 ? n - 1 : 0));
    assert(rhs.size() == n);

    std::vector<Real> d = rhs;

    for (std::size_t i = 1; i < n; ++i) {
      const Real m = a[i - 1] / b[i - 1];
      b[i] -= m * c[i - 1];
      d[i] -= m * d[i - 1];
    }

    x[n - 1] = d[n - 1] / b[n - 1];
    for (std::size_t k = n - 1; k-- > 0;) {
      x[k] = (d[k] - c[k] * x[k + 1]) / b[k];
    }
  }

  // ============================================================
  // 1D Casulli solver (BC: StageEta and VelocityU only; NO FluxQ)
  // ============================================================
#if defined(CASE_1D)
  void Solve1D_(fcfd::pdemodel::SVEcas<Real, PDEOptions>& pde)
  {
    auto const& mp = pde.GetModelParams();
    auto const bc = pde.BC();

    const bool left_is_eta = (bc.spec.left == fcfd::pdemodel::SubcriticalBC::StageEta);
    const bool right_is_eta = (bc.spec.right == fcfd::pdemodel::SubcriticalBC::StageEta);
    const bool left_is_u = (bc.spec.left == fcfd::pdemodel::SubcriticalBC::VelocityU);
    const bool right_is_u = (bc.spec.right == fcfd::pdemodel::SubcriticalBC::VelocityU);

    const Real etaL = left_is_eta ? bc.left_value : Real(0);
    const Real etaR = right_is_eta ? bc.right_value : Real(0);
    const Real uL = left_is_u ? bc.left_value : Real(0);
    const Real uR = right_is_u ? bc.right_value : Real(0);

    if (m_nx < 3) {
      throw std::logic_error("CasulliWrap 1D: nx must be >= 3");
    }
    const Real dt = this->m_methodProps.dt;
    const Real dx = this->m_methodProps.dx;
    if (!(dt > Real(0) && dx > Real(0))) {
      throw std::logic_error("CasulliWrap 1D: dt/dx must be > 0");
    }

    // -------------------------
    // Initial conditions (NEW)
    // -------------------------
    m_eta1d.resize(m_nx);
    for (std::size_t i = 0; i < m_nx; ++i) {
      const Real x = Real(i) * dx;
      m_eta1d[i] = m_eta_ic_1d ? m_eta_ic_1d(i, x) : pde.Eta0();
    }

    m_uface1d.resize(m_nx - 1);
    for (std::size_t f = 0; f < m_nx - 1; ++f) {
      const Real xf = (Real(f) + Real(0.5)) * dx;
      m_uface1d[f] = m_u_ic_1d ? m_u_ic_1d(f, xf) : pde.U0();
    }

    std::vector<Real> Hc(m_nx, Real(0));
    for (std::size_t i = 0; i < m_nx; ++i) {
      const Real x = Real(i) * dx;
      const Real h = pde.BottomAt(x);
      Hc[i] = pde.HfromEta(h, m_eta1d[i]);
    }

    for (int step = 0; step < this->m_methodProps.nSteps; ++step) {
      // predictor u* at faces
      std::vector<Real> u_star(m_nx - 1, Real(0));
      for (std::size_t f = 0; f < m_nx - 1; ++f) {
        const Real ui = m_uface1d[f];
        const Real uim = (f > 0) ? m_uface1d[f - 1] : m_uface1d[f];

        u_star[f] = ui * (Real(1) - (dt / dx) * (ui - uim)) - mp.g * (dt / dx) * (m_eta1d[f + 1] - m_eta1d[f]);
      }

      // face depths
      std::vector<Real> Hf(m_nx - 1, Real(0));
      for (std::size_t f = 0; f < m_nx - 1; ++f) {
        Hf[f] = Real(0.5) * (Hc[f] + Hc[f + 1]);
      }

      // gamma
      std::vector<Real> gamma(m_nx - 1, Real(0));
      for (std::size_t f = 0; f < m_nx - 1; ++f) {
        gamma[f] = pde.GammaFace(std::max(Hf[f], this->m_methodProps.dryTol), u_star[f]);
      }

      // alpha
      std::vector<Real> alpha(m_nx - 1, Real(0));
      for (std::size_t f = 0; f < m_nx - 1; ++f) {
        const Real denom = Hf[f] + gamma[f] * dt;
        if (Hf[f] <= this->m_methodProps.dryTol || denom <= this->m_methodProps.dryTol) {
          alpha[f] = Real(0);
        }
        else {
          alpha[f] = (Hf[f] * Hf[f]) / denom;
        }
      }

      // G
      std::vector<Real> G(m_nx - 1, Real(0));
      for (std::size_t f = 0; f < m_nx - 1; ++f) {
        G[f] = (Hf[f] * u_star[f] + dt * mp.gammat * pde.WindSpeedU());
      }

      // b starts with H^n
      std::vector<Real> b(m_nx, Real(0));
      for (std::size_t i = 0; i < m_nx; ++i) {
        b[i] = Hc[i];
      }

      auto HG_over_interior_face = [&](std::size_t f) -> Real
      {
        const Real denom = Hf[f] + gamma[f] * dt;
        if (Hf[f] <= this->m_methodProps.dryTol || denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        return (Hf[f] * G[f]) / denom;
      };

      auto HG_over_left_boundary = [&]() -> Real
      {
        if (!left_is_u) {
          return Real(0);
        }
        const Real H = Hc[0];
        if (H <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real gam = pde.GammaFace(std::max(H, this->m_methodProps.dryTol), uL);
        const Real denom = H + gam * dt;
        if (denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real Gbc = H * uL + dt * mp.gammat * pde.WindSpeedU();
        return (H * Gbc) / denom;
      };

      auto HG_over_right_boundary = [&]() -> Real
      {
        if (!right_is_u) {
          return Real(0);
        }
        const Real H = Hc[m_nx - 1];
        if (H <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real gam = pde.GammaFace(std::max(H, this->m_methodProps.dryTol), uR);
        const Real denom = H + gam * dt;
        if (denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real Gbc = H * uR + dt * mp.gammat * pde.WindSpeedU();
        return (H * Gbc) / denom;
      };

      // divergence everywhere
      for (std::size_t i = 0; i < m_nx; ++i) {
        Real Fp = Real(0), Fm = Real(0);

        if (i < m_nx - 1) {
          Fp = HG_over_interior_face(i);
        }
        else {
          Fp = HG_over_right_boundary();
        }

        if (i > 0) {
          Fm = HG_over_interior_face(i - 1);
        }
        else {
          Fm = HG_over_left_boundary();
        }

        b[i] -= (dt / dx) * (Fp - Fm);
      }

      // T tri-diagonal
      std::vector<Real> a(m_nx - 1, Real(0));
      std::vector<Real> d(m_nx, Real(0));
      std::vector<Real> c(m_nx - 1, Real(0));

      const Real fac = mp.g * (dt * dt) / (dx * dx);

      for (std::size_t i = 0; i < m_nx; ++i) {
        const Real ap = (i < m_nx - 1) ? alpha[i] : Real(0);
        const Real am = (i > 0) ? alpha[i - 1] : Real(0);
        d[i] = fac * (ap + am);
        if (i < m_nx - 1) {
          c[i] = -fac * ap;
        }
        if (i > 0) {
          a[i - 1] = -fac * am;
        }
      }

      // Newton on z
      std::vector<Real> z = m_eta1d;

      auto enforce_lower_bound = [&]()
      {
        for (std::size_t i = 0; i < m_nx; ++i) {
          const Real h = pde.BottomAt(Real(i) * dx);
          if (z[i] <= -h) {
            z[i] = -h + Real(1e-8);
          }
        }
      };

      enforce_lower_bound();

      if (left_is_eta) {
        z[0] = etaL;
      }
      if (right_is_eta) {
        z[m_nx - 1] = etaR;
      }

      for (int it = 0; it < this->m_methodProps.newtonMaxIts; ++it) {
        std::vector<Real> Hz(m_nx, Real(0));
        std::vector<Real> Ddiag(m_nx, Real(0));

        for (std::size_t i = 0; i < m_nx; ++i) {
          const Real h = pde.BottomAt(Real(i) * dx);
          const Real s = h + z[i];
          if (s > Real(0)) {
            Hz[i] = s;
            Ddiag[i] = Real(1);
          }
          else {
            Hz[i] = Real(0);
            Ddiag[i] = Real(0);
          }
        }

        std::vector<Real> r(m_nx, Real(0));
        for (std::size_t i = 0; i < m_nx; ++i) {
          Real Tz = d[i] * z[i];
          if (i > 0) {
            Tz += a[i - 1] * z[i - 1];
          }
          if (i + 1 < m_nx) {
            Tz += c[i] * z[i + 1];
          }
          r[i] = Hz[i] + Tz - b[i];
        }

        // Dirichlet residuals
        if (left_is_eta) {
          r[0] = z[0] - etaL;
        }
        if (right_is_eta) {
          r[m_nx - 1] = z[m_nx - 1] - etaR;
        }

        const Real rinf = NormInf_(r);
        if (m_newtonVerbose) {
          std::cout << "[1D step " << step << "] Newton it=" << it << "  ||r||_inf=" << std::scientific
                    << std::setprecision(6) << rinf << std::defaultfloat << "\n";
        }

        if (rinf < this->m_methodProps.newtonTol) {
          if (m_newtonVerbose) {
            std::cout << "[1D step " << step << "] Newton converged in " << it << " iterations\n";
          }
          break;
        }

        std::vector<Real> dd = d;
        for (std::size_t i = 0; i < m_nx; ++i) {
          dd[i] += Ddiag[i];
        }

        std::vector<Real> aa = a;
        std::vector<Real> cc = c;

        // Identity rows for Dirichlet
        if (left_is_eta) {
          dd[0] = Real(1);
          if (m_nx > 1) {
            cc[0] = Real(0);
          }
        }
        if (right_is_eta) {
          dd[m_nx - 1] = Real(1);
          if (m_nx > 1) {
            aa[m_nx - 2] = Real(0);
          }
        }

        std::vector<Real> delta;
        ThomasSolve_(aa, dd, cc, r, delta);

        for (std::size_t i = 0; i < m_nx; ++i) {
          z[i] -= delta[i];
        }

        if (left_is_eta) {
          z[0] = etaL;
        }
        if (right_is_eta) {
          z[m_nx - 1] = etaR;
        }

        enforce_lower_bound();
      }

      m_eta1d = z;

      for (std::size_t i = 0; i < m_nx; ++i) {
        const Real h = pde.BottomAt(Real(i) * dx);
        Hc[i] = pde.HfromEta(h, m_eta1d[i]);
      }

      // recover u^{n+1}
      for (std::size_t f = 0; f < m_nx - 1; ++f) {
        const Real denom = Hf[f] + gamma[f] * dt;
        if (Hf[f] <= this->m_methodProps.dryTol || denom <= this->m_methodProps.dryTol) {
          m_uface1d[f] = Real(0);
          continue;
        }
        const Real dEta = (m_eta1d[f + 1] - m_eta1d[f]);
        const Real rhs_u = Hf[f] * u_star[f] - mp.g * dt * (Hf[f]) * (dEta / dx) + dt * mp.gammat * pde.WindSpeedU();
        m_uface1d[f] = rhs_u / denom;
      }

      if (left_is_u) {
        m_uface1d[0] = uL;
      }
      if (right_is_u) {
        m_uface1d[m_nx - 2] = uR;
      }

      for (std::size_t f = 0; f < m_nx - 1; ++f) {
        if (Hf[f] <= this->m_methodProps.dryTol) {
          m_uface1d[f] = Real(0);
        }
      }
    }
  }
#endif  // CASE_1D

  // ============================================================
  // 2D Casulli solver (BC: StageEta and Velocity only)
  // ============================================================
#if defined(CASE_2D)
  void Solve2D_(fcfd::pdemodel::SWE2dcas<Real, PDEOptions>& pde)
  {
    auto const& mp = pde.ModelParams();
    auto const bc = pde.BC();

    if (m_nx < 3 || m_ny < 3) {
      throw std::logic_error("CasulliWrap 2D: nx,ny must be >= 3");
    }

    const Real dt = this->m_methodProps.dt;
    const Real dx = this->m_methodProps.dx;
    const Real dy = this->m_methodProps.dy;
    if (!(dt > Real(0) && dx > Real(0) && dy > Real(0))) {
      throw std::logic_error("CasulliWrap 2D: dt/dx/dy must be > 0");
    }

    const std::size_t N = m_nx * m_ny;

    // -------------------------
    // Initial conditions (NEW)
    // -------------------------
    m_eta2d.resize(N);
    for (std::size_t j = 0; j < m_ny; ++j) {
      for (std::size_t i = 0; i < m_nx; ++i) {
        const Real x = Real(i) * dx;
        const Real y = Real(j) * dy;
        m_eta2d[Idx_(i, j, m_nx)] = m_eta_ic_2d ? m_eta_ic_2d(i, j, x, y) : pde.Eta0();
      }
    }

    std::vector<Real> Hc(N, Real(0));
    for (std::size_t j = 0; j < m_ny; ++j) {
      for (std::size_t i = 0; i < m_nx; ++i) {
        const Real x = Real(i) * dx;
        const Real y = Real(j) * dy;
        const Real h = pde.BottomAt(x, y);
        Hc[Idx_(i, j, m_nx)] = pde.HfromEta(h, m_eta2d[Idx_(i, j, m_nx)]);
      }
    }

    m_uface2d.assign((m_nx - 1) * m_ny, pde.U0());
    m_vface2d.assign(m_nx * (m_ny - 1), pde.V0());

    auto idxU = [&](std::size_t i, std::size_t j) -> std::size_t { return i + (m_nx - 1) * j; };
    auto idxV = [&](std::size_t i, std::size_t j) -> std::size_t { return i + m_nx * j; };

    const bool left_is_eta = (bc.left == fcfd::pdemodel::SubcriticalBC2D::StageEta);
    const bool right_is_eta = (bc.right == fcfd::pdemodel::SubcriticalBC2D::StageEta);
    const bool bottom_is_eta = (bc.bottom == fcfd::pdemodel::SubcriticalBC2D::StageEta);
    const bool top_is_eta = (bc.top == fcfd::pdemodel::SubcriticalBC2D::StageEta);

    const bool left_is_vel = (bc.left == fcfd::pdemodel::SubcriticalBC2D::Velocity);
    const bool right_is_vel = (bc.right == fcfd::pdemodel::SubcriticalBC2D::Velocity);
    const bool bottom_is_vel = (bc.bottom == fcfd::pdemodel::SubcriticalBC2D::Velocity);
    const bool top_is_vel = (bc.top == fcfd::pdemodel::SubcriticalBC2D::Velocity);

    const Real etaL = left_is_eta ? bc.left_value : Real(0);
    const Real etaR = right_is_eta ? bc.right_value : Real(0);
    const Real etaB = bottom_is_eta ? bc.bottom_value : Real(0);
    const Real etaT = top_is_eta ? bc.top_value : Real(0);

    const Real uL = left_is_vel ? bc.left_value : Real(0);
    const Real uR = right_is_vel ? bc.right_value : Real(0);
    const Real vB = bottom_is_vel ? bc.bottom_value : Real(0);
    const Real vT = top_is_vel ? bc.top_value : Real(0);

    for (int step = 0; step < this->m_methodProps.nSteps; ++step) {
      std::vector<Real> u_star(m_uface2d.size(), Real(0));
      std::vector<Real> v_star(m_vface2d.size(), Real(0));

      for (std::size_t j = 0; j < m_ny; ++j) {
        for (std::size_t i = 0; i < m_nx - 1; ++i) {
          const Real ui = m_uface2d[idxU(i, j)];
          const Real uim = (i > 0) ? m_uface2d[idxU(i - 1, j)] : ui;

          const Real vjp = (j < m_ny - 1 && j > 0) ? m_vface2d[idxV(i, j)] : Real(0);
          const Real vjm = (j > 0) ? m_vface2d[idxV(i, j - 1)] : vjp;

          const Real adv = (dt / dx) * (ui - uim) + (dt / dy) * (vjp - vjm);
          const Real deta = m_eta2d[Idx_(i + 1, j, m_nx)] - m_eta2d[Idx_(i, j, m_nx)];
          u_star[idxU(i, j)] = ui * (Real(1) - adv) - mp.g * (dt / dx) * deta;
        }
      }

      for (std::size_t j = 0; j < m_ny - 1; ++j) {
        for (std::size_t i = 0; i < m_nx; ++i) {
          const Real vi = m_vface2d[idxV(i, j)];
          const Real vjm = (j > 0) ? m_vface2d[idxV(i, j - 1)] : vi;

          const Real uip = (i < m_nx - 1 && i > 0) ? m_uface2d[idxU(i, j)] : Real(0);
          const Real uim = (i > 0) ? m_uface2d[idxU(i - 1, j)] : uip;

          const Real adv = (dt / dy) * (vi - vjm) + (dt / dx) * (uip - uim);
          const Real deta = m_eta2d[Idx_(i, j + 1, m_nx)] - m_eta2d[Idx_(i, j, m_nx)];
          v_star[idxV(i, j)] = vi * (Real(1) - adv) - mp.g * (dt / dy) * deta;
        }
      }

      std::vector<Real> Hx((m_nx - 1) * m_ny, Real(0));
      std::vector<Real> Hy(m_nx * (m_ny - 1), Real(0));

      for (std::size_t j = 0; j < m_ny; ++j) {
        for (std::size_t i = 0; i < m_nx - 1; ++i) {
          Hx[idxU(i, j)] = Real(0.5) * (Hc[Idx_(i, j, m_nx)] + Hc[Idx_(i + 1, j, m_nx)]);
        }
      }

      for (std::size_t j = 0; j < m_ny - 1; ++j) {
        for (std::size_t i = 0; i < m_nx; ++i) {
          Hy[idxV(i, j)] = Real(0.5) * (Hc[Idx_(i, j, m_nx)] + Hc[Idx_(i, j + 1, m_nx)]);
        }
      }

      std::vector<Real> gamU(Hx.size(), Real(0));
      std::vector<Real> gamV(Hy.size(), Real(0));
      for (std::size_t k = 0; k < Hx.size(); ++k) {
        gamU[k] = pde.GammaFace(std::max(Hx[k], this->m_methodProps.dryTol), std::abs(u_star[k]));
      }
      for (std::size_t k = 0; k < Hy.size(); ++k) {
        gamV[k] = pde.GammaFace(std::max(Hy[k], this->m_methodProps.dryTol), std::abs(v_star[k]));
      }

      std::vector<Real> ax(Hx.size(), Real(0));
      std::vector<Real> ay(Hy.size(), Real(0));
      for (std::size_t k = 0; k < Hx.size(); ++k) {
        const Real denom = Hx[k] + gamU[k] * dt;
        ax[k] = (Hx[k] > this->m_methodProps.dryTol && denom > this->m_methodProps.dryTol) ? (Hx[k] * Hx[k] / denom)
                                                                                           : Real(0);
      }
      for (std::size_t k = 0; k < Hy.size(); ++k) {
        const Real denom = Hy[k] + gamV[k] * dt;
        ay[k] = (Hy[k] > this->m_methodProps.dryTol && denom > this->m_methodProps.dryTol) ? (Hy[k] * Hy[k] / denom)
                                                                                           : Real(0);
      }

      std::vector<Real> b = Hc;

      auto HGx_over_interior = [&](std::size_t i, std::size_t j) -> Real
      {
        const std::size_t f = idxU(i, j);
        const Real denom = Hx[f] + gamU[f] * dt;
        if (Hx[f] <= this->m_methodProps.dryTol || denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real G = Hx[f] * u_star[f] + dt * mp.gammat * pde.WindU();
        return (Hx[f] * G) / denom;
      };
      auto HGy_over_interior = [&](std::size_t i, std::size_t j) -> Real
      {
        const std::size_t f = idxV(i, j);
        const Real denom = Hy[f] + gamV[f] * dt;
        if (Hy[f] <= this->m_methodProps.dryTol || denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real G = Hy[f] * v_star[f] + dt * mp.gammat * pde.WindV();
        return (Hy[f] * G) / denom;
      };

      auto HGx_over_left_boundary = [&](std::size_t j) -> Real
      {
        if (!left_is_vel) {
          return Real(0);
        }
        const Real H = Hc[Idx_(0, j, m_nx)];
        if (H <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real gam = pde.GammaFace(std::max(H, this->m_methodProps.dryTol), std::abs(uL));
        const Real denom = H + gam * dt;
        if (denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real Gbc = H * uL + dt * mp.gammat * pde.WindU();
        return (H * Gbc) / denom;
      };
      auto HGx_over_right_boundary = [&](std::size_t j) -> Real
      {
        if (!right_is_vel) {
          return Real(0);
        }
        const Real H = Hc[Idx_(m_nx - 1, j, m_nx)];
        if (H <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real gam = pde.GammaFace(std::max(H, this->m_methodProps.dryTol), std::abs(uR));
        const Real denom = H + gam * dt;
        if (denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real Gbc = H * uR + dt * mp.gammat * pde.WindU();
        return (H * Gbc) / denom;
      };

      auto HGy_over_bottom_boundary = [&](std::size_t i) -> Real
      {
        if (!bottom_is_vel) {
          return Real(0);
        }
        const Real H = Hc[Idx_(i, 0, m_nx)];
        if (H <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real gam = pde.GammaFace(std::max(H, this->m_methodProps.dryTol), std::abs(vB));
        const Real denom = H + gam * dt;
        if (denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real Gbc = H * vB + dt * mp.gammat * pde.WindV();
        return (H * Gbc) / denom;
      };
      auto HGy_over_top_boundary = [&](std::size_t i) -> Real
      {
        if (!top_is_vel) {
          return Real(0);
        }
        const Real H = Hc[Idx_(i, m_ny - 1, m_nx)];
        if (H <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real gam = pde.GammaFace(std::max(H, this->m_methodProps.dryTol), std::abs(vT));
        const Real denom = H + gam * dt;
        if (denom <= this->m_methodProps.dryTol) {
          return Real(0);
        }
        const Real Gbc = H * vT + dt * mp.gammat * pde.WindV();
        return (H * Gbc) / denom;
      };

      for (std::size_t j = 0; j < m_ny; ++j) {
        for (std::size_t i = 0; i < m_nx; ++i) {
          Real Fxp = Real(0), Fxm = Real(0);
          if (i < m_nx - 1) {
            Fxp = HGx_over_interior(i, j);
          }
          else {
            Fxp = HGx_over_right_boundary(j);
          }

          if (i > 0) {
            Fxm = HGx_over_interior(i - 1, j);
          }
          else {
            Fxm = HGx_over_left_boundary(j);
          }

          Real Fyp = Real(0), Fym = Real(0);
          if (j < m_ny - 1) {
            Fyp = HGy_over_interior(i, j);
          }
          else {
            Fyp = HGy_over_top_boundary(i);
          }

          if (j > 0) {
            Fym = HGy_over_interior(i, j - 1);
          }
          else {
            Fym = HGy_over_bottom_boundary(i);
          }

          b[Idx_(i, j, m_nx)] -= (dt / dx) * (Fxp - Fxm) + (dt / dy) * (Fyp - Fym);
        }
      }

      std::vector<Real> z = m_eta2d;

      auto enforce_lower_bound = [&]()
      {
        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          for (std::size_t ii = 0; ii < m_nx; ++ii) {
            const Real x = Real(ii) * dx, y = Real(jj) * dy;
            const Real h = pde.BottomAt(x, y);
            auto& zij = z[Idx_(ii, jj, m_nx)];
            if (zij <= -h) {
              zij = -h + Real(1e-8);
            }
          }
        }
      };

      enforce_lower_bound();

      auto is_dirichlet = [&](std::size_t i, std::size_t j) -> bool
      {
        if (left_is_eta && i == 0) {
          return true;
        }
        if (right_is_eta && i == m_nx - 1) {
          return true;
        }
        if (bottom_is_eta && j == 0) {
          return true;
        }
        if (top_is_eta && j == m_ny - 1) {
          return true;
        }
        return false;
      };
      auto eta_bc = [&](std::size_t i, std::size_t j) -> Real
      {
        if (left_is_eta && i == 0) {
          return etaL;
        }
        if (right_is_eta && i == m_nx - 1) {
          return etaR;
        }
        if (bottom_is_eta && j == 0) {
          return etaB;
        }
        if (top_is_eta && j == m_ny - 1) {
          return etaT;
        }
        return Real(0);
      };

      for (std::size_t jj = 0; jj < m_ny; ++jj) {
        for (std::size_t ii = 0; ii < m_nx; ++ii) {
          if (is_dirichlet(ii, jj)) {
            z[Idx_(ii, jj, m_nx)] = eta_bc(ii, jj);
          }
        }
      }

      const Real facx = mp.g * (dt * dt) / (dx * dx);
      const Real facy = mp.g * (dt * dt) / (dy * dy);

      auto apply_T = [&](std::vector<Real> const& x, std::vector<Real>& Tx)
      {
        Tx.assign(N, Real(0));
        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          for (std::size_t ii = 0; ii < m_nx; ++ii) {
            if (is_dirichlet(ii, jj)) {
              Tx[Idx_(ii, jj, m_nx)] = Real(0);
              continue;
            }

            Real center = Real(0);

            if (ii + 1 < m_nx) {
              const Real ap = (ii < m_nx - 1) ? ax[idxU(ii, jj)] : Real(0);
              center += facx * ap * (x[Idx_(ii, jj, m_nx)] - x[Idx_(ii + 1, jj, m_nx)]);
            }
            if (ii > 0) {
              const Real am = ax[idxU(ii - 1, jj)];
              center += facx * am * (x[Idx_(ii, jj, m_nx)] - x[Idx_(ii - 1, jj, m_nx)]);
            }

            if (jj + 1 < m_ny) {
              const Real ap = (jj < m_ny - 1) ? ay[idxV(ii, jj)] : Real(0);
              center += facy * ap * (x[Idx_(ii, jj, m_nx)] - x[Idx_(ii, jj + 1, m_nx)]);
            }
            if (jj > 0) {
              const Real am = ay[idxV(ii, jj - 1)];
              center += facy * am * (x[Idx_(ii, jj, m_nx)] - x[Idx_(ii, jj - 1, m_nx)]);
            }

            Tx[Idx_(ii, jj, m_nx)] = center;
          }
        }
      };

      auto cg_solve = [&](std::vector<Real> const& Ddiag, std::vector<Real> const& rhs, std::vector<Real>& x)
      {
        x.assign(N, Real(0));

        std::vector<Real> r = rhs;

        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          for (std::size_t ii = 0; ii < m_nx; ++ii) {
            if (is_dirichlet(ii, jj)) {
              x[Idx_(ii, jj, m_nx)] = r[Idx_(ii, jj, m_nx)];
            }
          }
        }

        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          for (std::size_t ii = 0; ii < m_nx; ++ii) {
            if (is_dirichlet(ii, jj)) {
              r[Idx_(ii, jj, m_nx)] = Real(0);
            }
          }
        }

        std::vector<Real> pvec = r;
        std::vector<Real> Ap(N, Real(0));

        auto dot = [&](std::vector<Real> const& a, std::vector<Real> const& b) -> Real
        {
          Real s = Real(0);
          for (std::size_t k = 0; k < N; ++k) {
            s += a[k] * b[k];
          }
          return s;
        };

        Real rr = dot(r, r);
        if (std::sqrt(rr) < this->m_methodProps.cgTol) {
          return;
        }

        for (int it = 0; it < this->m_methodProps.cgMaxIts; ++it) {
          std::vector<Real> Tp;
          apply_T(pvec, Tp);

          for (std::size_t k = 0; k < N; ++k) {
            Ap[k] = Tp[k] + Ddiag[k] * pvec[k];
          }

          const Real denom = std::max(dot(pvec, Ap), Real(1e-30));
          const Real alpha = rr / denom;

          for (std::size_t k = 0; k < N; ++k) {
            x[k] += alpha * pvec[k];
          }
          for (std::size_t k = 0; k < N; ++k) {
            r[k] -= alpha * Ap[k];
          }

          const Real rr_new = dot(r, r);
          if (std::sqrt(rr_new) < this->m_methodProps.cgTol) {
            break;
          }

          const Real beta = rr_new / std::max(rr, Real(1e-30));
          for (std::size_t k = 0; k < N; ++k) {
            pvec[k] = r[k] + beta * pvec[k];
          }
          rr = rr_new;
        }
      };

      for (int it = 0; it < this->m_methodProps.newtonMaxIts; ++it) {
        std::vector<Real> Hz(N, Real(0));
        std::vector<Real> Ddiag(N, Real(0));
        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          for (std::size_t ii = 0; ii < m_nx; ++ii) {
            const Real x = Real(ii) * dx, y = Real(jj) * dy;
            const Real h = pde.BottomAt(x, y);
            const Real s = h + z[Idx_(ii, jj, m_nx)];
            if (s > Real(0)) {
              Hz[Idx_(ii, jj, m_nx)] = s;
              Ddiag[Idx_(ii, jj, m_nx)] = Real(1);
            }
            else {
              Hz[Idx_(ii, jj, m_nx)] = Real(0);
              Ddiag[Idx_(ii, jj, m_nx)] = Real(0);
            }
          }
        }

        std::vector<Real> Tz;
        apply_T(z, Tz);

        std::vector<Real> r(N, Real(0));
        for (std::size_t k = 0; k < N; ++k) {
          r[k] = Hz[k] + Tz[k] - b[k];
        }

        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          for (std::size_t ii = 0; ii < m_nx; ++ii) {
            if (is_dirichlet(ii, jj)) {
              r[Idx_(ii, jj, m_nx)] = z[Idx_(ii, jj, m_nx)] - eta_bc(ii, jj);
            }
          }
        }

        const Real rinf = NormInf_(r);
        if (m_newtonVerbose) {
          std::cout << "[2D step " << step << "] Newton it=" << it << "  ||r||_inf=" << std::scientific
                    << std::setprecision(6) << rinf << std::defaultfloat << "\n";
        }

        if (rinf < this->m_methodProps.newtonTol) {
          if (m_newtonVerbose) {
            std::cout << "[2D step " << step << "] Newton converged in " << it << " iterations\n";
          }
          break;
        }

        std::vector<Real> delta;
        cg_solve(Ddiag, r, delta);

        for (std::size_t k = 0; k < N; ++k) {
          z[k] -= delta[k];
        }

        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          for (std::size_t ii = 0; ii < m_nx; ++ii) {
            if (is_dirichlet(ii, jj)) {
              z[Idx_(ii, jj, m_nx)] = eta_bc(ii, jj);
            }
          }
        }

        enforce_lower_bound();
      }

      m_eta2d = z;

      for (std::size_t jj = 0; jj < m_ny; ++jj) {
        for (std::size_t ii = 0; ii < m_nx; ++ii) {
          const Real x = Real(ii) * dx;
          const Real y = Real(jj) * dy;
          const Real h = pde.BottomAt(x, y);
          Hc[Idx_(ii, jj, m_nx)] = pde.HfromEta(h, m_eta2d[Idx_(ii, jj, m_nx)]);
        }
      }

      for (std::size_t jj = 0; jj < m_ny; ++jj) {
        for (std::size_t ii = 0; ii < m_nx - 1; ++ii) {
          const auto f = idxU(ii, jj);
          const Real denom = Hx[f] + gamU[f] * dt;
          if (Hx[f] <= this->m_methodProps.dryTol || denom <= this->m_methodProps.dryTol) {
            m_uface2d[f] = Real(0);
            continue;
          }
          const Real dEta = m_eta2d[Idx_(ii + 1, jj, m_nx)] - m_eta2d[Idx_(ii, jj, m_nx)];
          const Real rhs_u = Hx[f] * u_star[f] - mp.g * dt * (Hx[f]) * (dEta / dx) + dt * mp.gammat * pde.WindU();
          m_uface2d[f] = rhs_u / denom;
        }
      }

      for (std::size_t jj = 0; jj < m_ny - 1; ++jj) {
        for (std::size_t ii = 0; ii < m_nx; ++ii) {
          const auto f = idxV(ii, jj);
          const Real denom = Hy[f] + gamV[f] * dt;
          if (Hy[f] <= this->m_methodProps.dryTol || denom <= this->m_methodProps.dryTol) {
            m_vface2d[f] = Real(0);
            continue;
          }
          const Real dEta = m_eta2d[Idx_(ii, jj + 1, m_nx)] - m_eta2d[Idx_(ii, jj, m_nx)];
          const Real rhs_v = Hy[f] * v_star[f] - mp.g * dt * (Hy[f]) * (dEta / dy) + dt * mp.gammat * pde.WindV();
          m_vface2d[f] = rhs_v / denom;
        }
      }

      if (left_is_vel) {
        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          m_uface2d[idxU(0, jj)] = uL;
        }
      }
      if (right_is_vel) {
        for (std::size_t jj = 0; jj < m_ny; ++jj) {
          m_uface2d[idxU(m_nx - 2, jj)] = uR;
        }
      }
      if (bottom_is_vel) {
        for (std::size_t ii = 0; ii < m_nx; ++ii) {
          m_vface2d[idxV(ii, 0)] = vB;
        }
      }
      if (top_is_vel) {
        for (std::size_t ii = 0; ii < m_nx; ++ii) {
          m_vface2d[idxV(ii, m_ny - 2)] = vT;
        }
      }

      for (std::size_t k = 0; k < Hx.size(); ++k) {
        if (Hx[k] <= this->m_methodProps.dryTol) {
          m_uface2d[k] = Real(0);
        }
      }
      for (std::size_t k = 0; k < Hy.size(); ++k) {
        if (Hy[k] <= this->m_methodProps.dryTol) {
          m_vface2d[k] = Real(0);
        }
      }
    }
  }
#endif  // CASE_2D

private:
  std::size_t m_nx {0};
  std::size_t m_ny {0};

  std::vector<Real> m_eta1d;
  std::vector<Real> m_uface1d;

  std::vector<Real> m_eta2d;
  std::vector<Real> m_uface2d;
  std::vector<Real> m_vface2d;

  // NEW: IC functors
  std::function<Real(std::size_t, Real)> m_eta_ic_1d {};
  std::function<Real(std::size_t, Real)> m_u_ic_1d {};

#if defined(CASE_2D)
  std::function<Real(std::size_t, std::size_t, Real, Real)> m_eta_ic_2d {};
#endif

  // NEW: Newton logging
  bool m_newtonVerbose {false};
};

}  // namespace fcfd::pdenumerics
