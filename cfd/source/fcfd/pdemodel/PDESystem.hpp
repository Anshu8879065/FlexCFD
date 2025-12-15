#pragma once

#include <cassert>
#include <concepts>
#include <functional>
#include <map>
#include <span>
#include <vector>

#include "BoundaryCondition.hpp"
#include "PDEParams.hpp"
#include "PDEType.hpp"

namespace fcfd::pdemodel
{

template<std::floating_point FloatingPointType, typename PDEOptions>
class PDESystem
{
public:
  using SystemFunctionType =
    std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType>)>;

  ///
  /// \brief Set the bottom for the PDESystem instance
  /// The "bottom" is the bed/bathymetry (the seabed or channel bed) that sets the elevation of the
  /// ground under the water. It essentially defines the bed elevation z(x) (or z(x, y)) used to
  /// compute the water depth and bed-slope source terms
  /// \param[in] seaFloor The value to be set as the bottom
  ///
  void SetBottom(
    std::function<std::vector<FloatingPointType>(std::span<const FloatingPointType>)> const&
      seaFloor)
  {
    m_groundFun = seaFloor;
    m_bottomSet = true;
  }

  ///
  /// \brief Set the bottom for the PDESystem instance
  ///
  virtual void SetBottom()
  {
  }

  ///
  /// \brief Initialize the PDE system
  /// This function initializes the PDE system by passing in a set of PDEOptions to be used in the
  /// evaluation of the PDE system
  /// \param[in] options The PDEOptions for the system
  ///
  void InitPDESystem(PDEOptions const& options)
  {
    m_myPdeOpts = options;
  }

  ///
  /// \brief Set the initial conditions function
  /// This function sets the initial conditions function for the PDESystem instance and then toggles
  /// a boolean condition to mark it as having been set
  /// \param[in] initConds The initial conditions function
  ///
  void SetInitCond(
    std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType> const&)> const&
      initConds)
  {
    m_myInitConds = initConds;
    m_initCondSet = true;
  }

  ///
  /// \brief Set the initial conditions function
  ///
  virtual void SetInitCond()
  {
  }

  ///
  /// \brief Pure virtual functions for initial conditions evaluation
  ///
  virtual auto InitCond() -> FloatingPointType = 0;

  ///
  /// \brief Set boundary conditions
  /// \param[in] bdryConds The boundary conditions to set
  /// \param[in] numConds The number of conditions to be set
  ///
  void SetBdryCond(std::map<int, BoundaryCondition<FloatingPointType>>&& bdryConds, int numConds)
  {
    m_myBdry = std::move(bdryConds);
    m_nbound = numConds;
    m_bdryCondSet = true;
  }

  ///
  /// \brief Set boundary conditions
  ///
  virtual void SetBdryCond()
  {
  }

  ///
  /// \brief Pure virtual function for boundary conditions evaluation
  /// \returns
  ///
  virtual auto BdryCond() -> FloatingPointType = 0;

  // RHS now takes an input vector and returns an output vector.
  void RHS(const std::vector<FloatingPointType>* x, std::vector<FloatingPointType>& out)
  {
    assert(m_boolRHSSet && "RHS function has not been set!");
    out = MyRHS(x);
  }

  // Jacobian RHS operator.
  void JacRHS(const std::vector<FloatingPointType>* x, std::vector<FloatingPointType>& out, int ro)
  {
    assert(m_boolJacRHSSet && "JacRHS function has not been set!");
    out = MyJacRHS(x, ro);
  }

  // LHS operator.
  void LHSOpAsplit(const std::vector<FloatingPointType>* v,
    const std::vector<FloatingPointType>* vdot,
    const std::vector<FloatingPointType>* dvdx,
    const std::vector<FloatingPointType>* dv2dx,
    const std::vector<FloatingPointType>* dv3dx,
    std::vector<FloatingPointType>& out)
  {
    assert(m_boolLHSOpSet && "LHSOp function has not been set!");
    out = MyLHSAsplit(v, vdot, dvdx, dv2dx, dv3dx);
  }

  void LHSOpBsplit(const std::vector<FloatingPointType>* v,
    const std::vector<FloatingPointType>* vdot,
    const std::vector<FloatingPointType>* dvdx,
    const std::vector<FloatingPointType>* dv2dx,
    const std::vector<FloatingPointType>* dv3dx,
    std::vector<FloatingPointType>& out)
  {
    assert(m_boolLHSOpSet && "LHSOp function has not been set!");
    out = MyLHSBsplit(v, vdot, dvdx, dv2dx, dv3dx);
  }

  // Jacobian LHS operator.
  void JacLHSOpAsplit(const std::vector<FloatingPointType>* v,
    const std::vector<FloatingPointType>* vdot,
    const std::vector<FloatingPointType>* dvdx,
    const std::vector<FloatingPointType>* dv2dx,
    const std::vector<FloatingPointType>* dv3dx,
    std::vector<FloatingPointType>* out,
    int rowo,
    int colo,
    int derivo)
  {
    assert(m_boolJacLHSOpSet && "JacLHSOp function has not been set!");
    out = MyJacLHSAsplit(v, vdot, dvdx, dv2dx, dv3dx, rowo, colo, derivo);
  }

  void JacLHSOpBsplit(const std::vector<FloatingPointType>& v,
    const std::vector<FloatingPointType>& vdot,
    const std::vector<FloatingPointType>& dvdx,
    const std::vector<FloatingPointType>& dv2dx,
    const std::vector<FloatingPointType>& dv3dx,
    std::vector<FloatingPointType>& out,
    int rowo,
    int colo,
    int derivo)
  {
    assert(m_boolJacLHSOpSet && "JacLHSOp function has not been set!");
    out = MyJacLHSBsplit(v, vdot, dvdx, dv2dx, dv3dx, rowo, colo, derivo);
  }

  ///
  /// \brief Setter for the RHS function
  /// \param[in] rhsConds The value to set the RHS function to
  ///
  void SetRHSFunc(std::function<
    auto(std::vector<FloatingPointType> const&)->std::vector<FloatingPointType>> const& rhsConds)
  {
    m_myRHS = rhsConds;
    m_boolRHSSet = true;
  }

  ///
  /// \brief Setter for the RHS function
  ///
  virtual void SetRHSFunc()
  {
  }

  ///
  /// \brief Setter for the Jacobian RHS function
  /// \param[in] rhsJacobianFunc The RHS Jacobian function
  ///
  void SetJacRHS(std::function<auto(std::vector<FloatingPointType> const&, int)
      ->std::vector<FloatingPointType>> const& rhsJacobianFunc)
  {
    m_myJacRHS = rhsJacobianFunc;
    m_boolJacRHSSet = true;
  }

  ///
  /// \brief Setter for the Jacobian RHS function
  ///
  virtual void SetJacRHS()
  {
  }

  // Setter for the LHS operator function. A/B split for implicit/explicit
  // distinguishing
  void SetLHSOp(const std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType>*,
                  std::vector<FloatingPointType>*,
                  std::vector<FloatingPointType>*,
                  std::vector<FloatingPointType>*,
                  std::vector<FloatingPointType>*)>& lhsa,
    const std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*)>& lhsb)
  {
    m_myLHSAsplit = lhsa;
    m_myLHSBsplit = lhsb;

    m_boolLHSOpSet = true;
  }

  virtual void SetLHSOp() = 0;

  // Setter for the Jacobian LHS operator function.

  void SetJacLHSOp(
    const std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      int,
      int,
      int)>& jaclhsa,
    const std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      std::vector<FloatingPointType>*,
      int,
      int,
      int)>& jaclhsb)
  {
    m_myJacLHSAsplit = jaclhsa;
    m_myJacLHSBsplit = jaclhsb;
    m_boolJacLHSOpSet = true;
  }

  virtual void SetJacLHSOp() = 0;

  ///
  /// \brief Initialize the PDE system.
  /// This function checks to see if the RHS, LHS, RHSJacobian, and LHSJacobian functions have been
  /// initialized, and sets them to default values if not
  ///
  void InitPDESys()
  {
    // If the RHS function is not set, provide a default function that returns a zero vector

    if (!m_boolRHSSet) {
      SetRHSFunc([](std::vector<FloatingPointType> const& rhs) -> std::vector<FloatingPointType>
        { return std::vector<FloatingPointType>(rhs.size(), static_cast<FloatingPointType>(0)); });
    }

    // If the Jacobian RHS function is not set, provide a default function.

    if (!m_boolJacRHSSet) {
      SetJacRHS(
        [](std::vector<FloatingPointType> const& rhsJac) -> std::vector<FloatingPointType>
        {
          return std::vector<FloatingPointType>(rhsJac.size(), static_cast<FloatingPointType>(0));
        });
    }

    // If the LHS operator function is not set, provide a default function.

    if (!m_boolLHSOpSet) {
      SetLHSOp([](std::vector<FloatingPointType> const& lhs) -> std::vector<FloatingPointType>
        { return std::vector<FloatingPointType>(lhs.size(), static_cast<FloatingPointType>(0)); });
    }

    // If the Jacobian LHS operator function is not set, provide a default function.

    if (!m_boolJacLHSOpSet) {
      SetJacLHSOp(
        [](std::vector<FloatingPointType> const& lhsJac) -> std::vector<FloatingPointType>
        {
          return std::vector<FloatingPointType>(lhsJac.size(), static_cast<FloatingPointType>(0));
        });
    }
  }

  // EvalSol calls the private 'Solution' function.
  void EvalSol(const std::vector<FloatingPointType>& x, std::vector<FloatingPointType>& out)
  {
    assert(m_solFound && "Solution function has not been set!");
    out = Solution(x);
  }

  ///
  /// \brief Copy constructor
  ///
  PDESystem(PDESystem const&) = default;

  ///
  /// \brief Copy-assignment operator
  ///
  auto operator=(PDESystem const&) -> PDESystem& = default;

  ///
  /// \brief Move constructor
  ///
  PDESystem(PDESystem&&) noexcept = default;

  ///
  /// \brief Move-assignment operator
  ///
  auto operator=(PDESystem&&) noexcept -> PDESystem& = default;

  ///
  /// \brief Virtual destructor
  ///
  virtual ~PDESystem() = default;

protected:
  ///
  /// \brief Constructor
  /// Creates an instance of the PDESystem class with the given PDEType and PDEParams values
  /// \param[in] pdeType The type of the PDE (either SVE, SWE2d, SWE3d, or Unknown)
  /// \param[in] params The parameters of the PDE
  ///
  PDESystem(PDEType pdeType, PDEParams<PDEOptions> const& params)
    : m_pdeType(pdeType)
    , m_pdeParams(params)
  {
  }

  bool m_bottomSet = false;
  bool m_initCondSet = false;
  bool m_bdryCondSet = false;
  bool m_paramsSet = false;
  bool m_gridParamsSet = false;

  bool m_boolRHSSet = false;
  bool m_boolJacRHSSet = false;
  bool m_boolLHSOpSet = false;
  bool m_boolJacLHSOpSet = false;
  bool m_solFound = false;

  SystemFunctionType m_groundFun;

  // Note: pdetype and nd are declared only once.
  PDEType m_pdeType;
  PDEOptions m_myPdeOpts;
  PDEParams<PDEOptions> m_pdeParams;
  int m_nd {};
  int m_nv {};
  int m_nbound {};

  std::map<int, BoundaryCondition<FloatingPointType>> m_myBdry;
  const std::map<int, BoundaryCondition<FloatingPointType>> m_defaultBoundary;
  SystemFunctionType m_defaultInitconds;
  SystemFunctionType m_myInitConds;
  SystemFunctionType m_myLHSAsplit;
  SystemFunctionType m_myJacLHSAsplit;
  SystemFunctionType m_myLHSBsplit;
  SystemFunctionType m_myJacLHSBsplit;
  SystemFunctionType m_myRHS;
  SystemFunctionType m_myJacRHS;
  SystemFunctionType m_solution;
};
}  // namespace fcfd::pdemodel
