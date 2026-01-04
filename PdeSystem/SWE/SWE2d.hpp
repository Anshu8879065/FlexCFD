#pragma once

#include <cmath>          // std::exp, std::sqrt
#include <concepts>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "BoundaryCondition.hpp"
#include "ModelParams.hpp"
#include "PDEParams.hpp"
#include "PDESystem.hpp"
#include "PDEType.hpp"

namespace fcfd::pdemodel
{

// ============================================================================
// SWE2d (2D depth-averaged shallow water, conservative variables)
// State variables chosen as:
//   U = [ h, qx, qy ]^T  where qx = U h, qy = V h
//
// Continuity:
//   h_t + (qx)_x + (qy)_y = 0
//
// Momentum-x:
//   (qx)_t + ( (qx^2)/h )_x + ( (qx*qy)/h )_y = - g h (Zs)_x - c_f U sqrt(U^2+V^2)
//
// Momentum-y:
//   (qy)_t + ( (qx*qy)/h )_x + ( (qy^2)/h )_y = - g h (Zs)_y - c_f V sqrt(U^2+V^2)
//
// Zs = Zb + h. Since PDESystem::RHS does not get (x,y), you supply (Zs)_x,(Zs)_y via members.
// ============================================================================

template<std::floating_point FloatingPointType, typename PDEOptions>
class SWE2d final : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using Base = PDESystem<FloatingPointType, PDEOptions>;
  using Base::InitPDESys;

  SWE2d()
    : Base(PDEType::SWE2d, PDEParams<PDEOptions>(2, 3))
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

  explicit SWE2d(Model2dParams<FloatingPointType> const& modelParams)
    : Base(PDEType::SWE2d, PDEParams<PDEOptions>(2, 3))
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

  auto GetModelParams() const noexcept -> Model2dParams<FloatingPointType>
  {
    return m_modelParams;
  }

  // -----------------------------
  // SWE2d: Initial condition
  // x[0]=x, x[1]=y
  // returns U0(x,y) = [h, qx, qy]
  // -----------------------------
  void SetInitCond() override
  {
    this->Base::SetInitCond(
      [](std::vector<FloatingPointType> const& x) -> std::vector<FloatingPointType>
      {
        const FloatingPointType xc = FloatingPointType(25.0);
        const FloatingPointType yc = FloatingPointType(25.0);

        const FloatingPointType dx = x[0] - xc;
        const FloatingPointType dy = x[1] - yc;
        const FloatingPointType dist_sq = dx * dx + dy * dy;

        const FloatingPointType h =
          FloatingPointType(5.0) + FloatingPointType(1.0) * std::exp(-dist_sq / FloatingPointType(10.0));

        return std::vector<FloatingPointType>{
          h,                    // h(x,y,0)
          FloatingPointType(0), // qx(x,y,0)
          FloatingPointType(0)  // qy(x,y,0)
        };
      });
  }

  auto InitCond() -> FloatingPointType override { return FloatingPointType{}; }

  // -----------------------------
  // SWE2d: Boundary conditions (transmissive / zero-gradient)
  // IDs below use:
  //   1 = left, 2 = right, 3 = bottom, 4 = top
  // This callback returns the interior state unchanged,
  // which wrapper can use as U_ghost = U_inner.
  // -----------------------------
  void SetBdryCond() override
  {
    std::unordered_map<int, BoundaryCondition<FloatingPointType>> bcs;

    auto transmissive =
      [](std::vector<FloatingPointType> const& u_inner, auto /*t*/) -> std::vector<FloatingPointType>
    {
      return u_inner; // zero-gradient / copy-through
    };

    bcs[1] = BoundaryCondition<FloatingPointType>(BoundaryType::Neumann, transmissive); // left
    bcs[2] = BoundaryCondition<FloatingPointType>(BoundaryType::Neumann, transmissive); // right
    bcs[3] = BoundaryCondition<FloatingPointType>(BoundaryType::Neumann, transmissive); // bottom
    bcs[4] = BoundaryCondition<FloatingPointType>(BoundaryType::Neumann, transmissive); // top

    this->Base::SetBdryCond(std::move(bcs), /*numConds=*/4);
  }

  auto BdryCond() -> FloatingPointType override { return FloatingPointType{}; }

  void SetBottom() override
  {
    Base::SetBottom(
      [](std::span<const FloatingPointType> /*coords*/) -> std::vector<FloatingPointType>
      {
        return {FloatingPointType(0)};
      });
  }

  void SetRHSFunc() override
  {
    Base::SetRHSFunc(
      [this](std::vector<FloatingPointType> const& input,
             std::vector<FloatingPointType>& out,
             int /*field*/) -> std::vector<FloatingPointType>
      {
        if (out.size() != 3) out.resize(3);

        const FloatingPointType h  = input[0];
        const FloatingPointType qx = input[1];
        const FloatingPointType qy = input[2];

        const FloatingPointType g   = m_modelParams.g;
        const FloatingPointType cf  = m_cf;    // friction coefficient
        const FloatingPointType rho = m_rho;   // density (kept explicit to match your equation)

        out[0] = FloatingPointType(0);

        if (h <= m_dryTol) {
          out[1] = FloatingPointType(0);
          out[2] = FloatingPointType(0);
          return out;
        }

        const FloatingPointType U = qx / h;
        const FloatingPointType V = qy / h;
        const FloatingPointType speed = std::sqrt(U * U + V * V);

        // (sigma_xz)_b = rho * cf * U * speed
        // (sigma_yz)_b = rho * cf * V * speed
        const FloatingPointType sig_xz_b = rho * cf * U * speed;
        const FloatingPointType sig_yz_b = rho * cf * V * speed;

        // - g h (Zs)_x - (1/rho) (sigma_xz)_b
        // - g h (Zs)_y - (1/rho) (sigma_yz)_b
        out[1] = -g * h * m_dZs_dx - (sig_xz_b / rho);
        out[2] = -g * h * m_dZs_dy - (sig_yz_b / rho);

        return out;
      });
  }

  void SetLHSOp() override
  {
    auto lhsa =
      [](std::vector<FloatingPointType> const& /*input*/,
         std::vector<FloatingPointType> const& vdot,
         std::vector<FloatingPointType> const& /*dvdx*/,
         std::optional<std::vector<FloatingPointType>> const& /*dvdy*/,
         std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
         std::vector<FloatingPointType>& out) -> std::vector<FloatingPointType>
    {
      if (out.size() != 3) out.resize(3);
      out[0] = vdot[0]; // dh/dt
      out[1] = vdot[1]; // d(qx)/dt
      out[2] = vdot[2]; // d(qy)/dt
      return out;
    };

    auto lhsb =
      [this](std::vector<FloatingPointType> const& input,
             std::vector<FloatingPointType> const& /*vdot*/,
             std::vector<FloatingPointType> const& dvdx,
             std::optional<std::vector<FloatingPointType>> const& dvdy,
             std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
             std::vector<FloatingPointType>& out) -> std::vector<FloatingPointType>
    {
      if (out.size() != 3) out.resize(3);

      const FloatingPointType h  = input[0];
      const FloatingPointType qx = input[1];
      const FloatingPointType qy = input[2];

      const FloatingPointType g = m_modelParams.g;

      // x-derivatives of state: [hx, qxx, qyx]
      const FloatingPointType hx  = dvdx[0];
      const FloatingPointType qxx = dvdx[1];
      const FloatingPointType qyx = dvdx[2];

      // y-derivatives of state: [hy, qxy, qyy]
      FloatingPointType hy  = FloatingPointType(0);
      FloatingPointType qxy = FloatingPointType(0);
      FloatingPointType qyy = FloatingPointType(0);
      if (dvdy.has_value()) {
        auto const& dy = dvdy.value();
        hy  = dy[0];
        qxy = dy[1];
        qyy = dy[2];
      }

      if (h <= m_dryTol) {
        out[0] = FloatingPointType(0);
        out[1] = FloatingPointType(0);
        out[2] = FloatingPointType(0);
        return out;
      }

      // Continuity: (qx)_x + (qy)_y
      out[0] = qxx + qyy;

      const FloatingPointType invh  = FloatingPointType(1) / h;
      const FloatingPointType invh2 = invh * invh;

      // Momentum-x: d/dx(qx^2/h + g h^2/2) + d/dy(qx*qy/h)
      const FloatingPointType dF2_x =
        -(qx * qx) * invh2 * hx + (FloatingPointType(2) * qx * invh) * qxx + (g * h) * hx;
      const FloatingPointType dG2_y =
        -(qx * qy) * invh2 * hy + (qy * invh) * qxy + (qx * invh) * qyy;

      out[1] = dF2_x + dG2_y;

      // Momentum-y: d/dx(qx*qy/h) + d/dy(qy^2/h + g h^2/2)
      const FloatingPointType dF3_x =
        -(qx * qy) * invh2 * hx + (qy * invh) * qxx + (qx * invh) * qyx;
      const FloatingPointType dG3_y =
        -(qy * qy) * invh2 * hy + (FloatingPointType(2) * qy * invh) * qyy + (g * h) * hy;

      out[2] = dF3_x + dG3_y;

      (void)g; // keeps “old” style; g used above, harmless if compiler warns in some configs
      return out;
    };

    Base::SetLHSOp(lhsa, lhsb);
  }

  // NOTE: This is kept exactly in the “old” style you pasted.
  // Your JacRHS/JacLHSOp interfaces are vector-based, so if you need a full 3x3,
  // handle row selection via 'field' in wrapper or pack/unpack convention.
  void SetJacLHSOp() override
  {
    auto jac_lhsa =
      [](std::vector<FloatingPointType> const& /*input*/,
         std::vector<FloatingPointType> const& /*vdot*/,
         std::vector<FloatingPointType> const& /*dvdx*/,
         std::optional<std::vector<FloatingPointType>> const& /*dv2dx*/,
         std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
         std::vector<FloatingPointType>& out,
         int rowo, int colo, int derivo) -> std::vector<FloatingPointType>
    {
      if (out.size() != 3) out.assign(3, FloatingPointType(0));

      if (derivo == 0) {
        out[0] = (rowo == 0 && colo == 0) ? FloatingPointType(1) : FloatingPointType(0);
        out[1] = (rowo == 1 && colo == 1) ? FloatingPointType(1) : FloatingPointType(0);
        out[2] = (rowo == 2 && colo == 2) ? FloatingPointType(1) : FloatingPointType(0);
      }
      return out;
    };

    auto jac_lhsb =
      [this](std::vector<FloatingPointType> const& input,
             std::vector<FloatingPointType> const& /*vdot*/,
             std::vector<FloatingPointType> const& /*dvdx*/,
             std::optional<std::vector<FloatingPointType>> const& /*dv2dx*/,
             std::optional<std::vector<FloatingPointType>> const& /*dv3dx*/,
             std::vector<FloatingPointType>& out,
             int rowo, int colo, int derivo) -> std::vector<FloatingPointType>
    {
      if (out.size() != 3) out.assign(3, FloatingPointType(0));

      const FloatingPointType h  = input[0];
      const FloatingPointType qx = input[1];
      const FloatingPointType qy = input[2];

      if (h <= m_dryTol) return out;

      const FloatingPointType g = m_modelParams.g;
      const FloatingPointType invh  = FloatingPointType(1) / h;
      const FloatingPointType invh2 = invh * invh;

      // derivo==1: coefficients for x-derivatives [hx, qxx, qyx]
      // derivo==2: coefficients for y-derivatives [hy, qxy, qyy]
      if (derivo == 1) {
        if (rowo == 0 && colo == 1) out[0] = FloatingPointType(1);

        if (rowo == 1 && colo == 0) out[1] = -(qx * qx) * invh2 + (g * h);
        if (rowo == 1 && colo == 1) out[1] = FloatingPointType(2) * qx * invh;

        if (rowo == 2 && colo == 0) out[2] = -(qx * qy) * invh2;
        if (rowo == 2 && colo == 1) out[2] = qy * invh;
        if (rowo == 2 && colo == 2) out[2] = qx * invh;
      }

      if (derivo == 2) {
        if (rowo == 0 && colo == 2) out[0] = FloatingPointType(1);

        if (rowo == 1 && colo == 0) out[1] = -(qx * qy) * invh2;
        if (rowo == 1 && colo == 1) out[1] = qy * invh;
        if (rowo == 1 && colo == 2) out[1] = qx * invh;

        if (rowo == 2 && colo == 0) out[2] = -(qy * qy) * invh2 + (g * h);
        if (rowo == 2 && colo == 2) out[2] = FloatingPointType(2) * qy * invh;
      }

      return out;
    };

    Base::SetJacLHSOp(jac_lhsa, jac_lhsb);
  }

  void SetJacRHS() override
  {
    Base::SetJacRHS(
      [this](std::vector<FloatingPointType> const& input,
             std::vector<FloatingPointType>& out,
             int /*field*/) -> std::vector<FloatingPointType>
      {
        if (out.size() != 3) out.assign(3, FloatingPointType(0));

        const FloatingPointType h  = input[0];
        const FloatingPointType qx = input[1];
        const FloatingPointType qy = input[2];

        if (h <= m_dryTol) return out;

        const FloatingPointType g  = m_modelParams.g;
        const FloatingPointType cf = m_cf;

        const FloatingPointType invh = FloatingPointType(1) / h;
        const FloatingPointType U = qx * invh;
        const FloatingPointType V = qy * invh;
        const FloatingPointType speed = std::sqrt(U * U + V * V);

        if (speed > FloatingPointType(0)) {
          const FloatingPointType dUsdU = speed + (U * U) / speed;
          const FloatingPointType dUsdV = (U * V) / speed;

          const FloatingPointType dU_dh  = -qx * invh * invh;
          const FloatingPointType dU_dqx = invh;

          const FloatingPointType dV_dh  = -qy * invh * invh;
          const FloatingPointType dV_dqy = invh;

          const FloatingPointType dFric2_dh  = dUsdU * dU_dh  + dUsdV * dV_dh;
          const FloatingPointType dFric2_dqx = dUsdU * dU_dqx;
          const FloatingPointType dFric2_dqy = dUsdV * dV_dqy;

          // This returns derivatives for the x-momentum source row.
          out[0] = -g * m_dZs_dx - cf * dFric2_dh;
          out[1] = -cf * dFric2_dqx;
          out[2] = -cf * dFric2_dqy;
        } else {
          out[0] = -g * m_dZs_dx;
          out[1] = FloatingPointType(0);
          out[2] = FloatingPointType(0);
        }

        return out;
      });
  }

private:
  Model2dParams<FloatingPointType> m_modelParams {};

  static constexpr Model2dParams<FloatingPointType> defaultModelParams {
    .g = FloatingPointType(9.8),
    .nu = FloatingPointType(1.0),
    .gammat = FloatingPointType(0.1),
    .gamman = FloatingPointType(0.8),
    .nm = FloatingPointType(0.03),
    .wsurf = FloatingPointType(2.0),
    .radi = FloatingPointType(0.7),
    .wsurfydir = FloatingPointType(1.5)
  };

  FloatingPointType m_cf     { FloatingPointType(0.03) };
  FloatingPointType m_rho    { FloatingPointType(1.0) };
  FloatingPointType m_dZs_dx { FloatingPointType(0.0) };
  FloatingPointType m_dZs_dy { FloatingPointType(0.0) };

  FloatingPointType m_dryTol { FloatingPointType(1e-12) };
};

} // namespace fcfd::pdemodel
