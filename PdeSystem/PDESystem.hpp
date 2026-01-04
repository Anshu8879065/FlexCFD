#pragma once

#include <cassert>
#include <concepts>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cmath>
#include <stdexcept>
#include <string>

#include "BoundaryCondition.hpp"
#include "PDEParams.hpp"
#include "PDEType.hpp"

namespace fcfd::pdemodel
{

template<std::floating_point FloatingPointType, std::semiregular PDEOptions>
class PDESystem
{
  using SystemFunctionType = std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType> const&)>;
  using SystemRHSFunctionType = std::function<std::vector<FloatingPointType>(
    std::vector<FloatingPointType> const&, std::vector<FloatingPointType>&, int)>;

  using SystemRHSJacFunctionType = std::function<std::vector<FloatingPointType>(
    std::vector<FloatingPointType> const&, std::vector<FloatingPointType>&, int)>;

  using SystemLHSFunctionType = std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType> const&,
    std::vector<FloatingPointType> const&,
    std::vector<FloatingPointType> const&,
    std::optional<std::vector<FloatingPointType>> const&,
    std::optional<std::vector<FloatingPointType>> const&,
    std::vector<FloatingPointType>&)>;

  using SystemLHSJacFunctionType = std::function<std::vector<FloatingPointType>(std::vector<FloatingPointType> const&,
    std::vector<FloatingPointType> const&,
    std::vector<FloatingPointType> const&,
    std::optional<std::vector<FloatingPointType>> const&,
    std::optional<std::vector<FloatingPointType>> const&,
    std::vector<FloatingPointType>&,
    int,
    int,
    int)>;

public:
  ///
  /// \brief Set the bottom for the PDESystem instance
  /// The "bottom" is the bed/bathymetry (the seabed or channel bed) that sets the elevation of the
  /// ground under the water. It essentially defines the bed elevation z(x) (or z(x, y)) used to
  /// compute the water depth and bed-slope source terms
  /// \param[in] seaFloor The value to be set as the bottom
  ///
  void SetBottom(std::function<std::vector<FloatingPointType>(std::span<const FloatingPointType>)> const& seaFloor)
  {
    m_groundFun = seaFloor;
    m_bottomSet = true;
  }

  ///
  /// \brief Set the bottom for the PDESystem instance
  ///
  virtual void SetBottom() = 0;

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
  void SetInitCond(SystemFunctionType const& initConds)
  {
    m_myInitConds = initConds;
    m_initCondSet = true;
  }

  ///
  /// \brief Set the initial conditions function
  ///
  virtual void SetInitCond() = 0;

  ///
  /// \brief Pure virtual functions for initial conditions evaluation
  ///
  virtual auto InitCond() -> FloatingPointType = 0;

  ///
  /// \brief Set boundary conditions
  /// \param[in] bdryConds The boundary conditions to set
  /// \param[in] numConds The number of conditions to be set
  ///
  void SetBdryCond(std::unordered_map<int, BoundaryCondition<FloatingPointType>>&& bdryConds, int numConds)
  {
    m_myBdry = std::move(bdryConds);
    m_nbound = numConds;
    m_bdryCondSet = true;
  }

  ///
  /// \brief Set boundary conditions
  ///
  virtual void SetBdryCond() = 0;

  ///
  /// \brief Pure virtual function for boundary conditions evaluation
  /// \returns
  ///
  virtual auto BdryCond() -> FloatingPointType = 0;

  ///
  /// \brief
  /// RHS now takes an input vector and returns an output vector.
  /// \param[in] input The input vector
  /// \param[in,out] output The output vector
  /// \param[in] field The current field being processed
  ///
  void RHS(std::vector<FloatingPointType> const& input, std::vector<FloatingPointType>& output, int field)
  {
    assert(m_boolRHSSet && "RHS function has not been set!");
    output = m_myRHS(input, output, field);
  }

  ///
  /// \brief Jacobian RHS operator
  /// \param[in] input The input vector
  /// \param[in,out] output The output vector
  /// \param[in] field The current field being processed
  ///
  void JacRHS(std::vector<FloatingPointType> const& input, std::vector<FloatingPointType>& output, int field)
  {
    assert(m_boolJacRHSSet && "JacRHS function has not been set!");
    output = m_myJacRHS(input, output, field);
  }

  ///
  /// \brief Compute the LHSAsplit for the implicit function
  /// \param[in] input A std::vector containing the field variables
  /// \param[in] vdot A std::vector containing the first-order time derivatives of the field variables
  /// \param[in] dvdx A std::vector containing the first-order space derivatives of the field variables
  /// \param[in] dv2dx A std::vector containing the second-order space derivatives of the field variables
  /// \param[in] dv3dx A std::vector containing the third-order space derivatives of the field variables
  /// \param[in,out] output A std::vector containing the computed residuals for the implicit function
  ///
  void LHSOpAsplit(std::vector<FloatingPointType> const& input,
    std::vector<FloatingPointType> const& vdot,
    std::vector<FloatingPointType> const& dvdx,
    std::optional<std::vector<FloatingPointType>> const& dv2dx,
    std::optional<std::vector<FloatingPointType>> const& dv3dx,
    std::vector<FloatingPointType>& output)
  {
    assert(m_boolLHSOpSet && "LHSOp function has not been set!");
    output = m_myLHSAsplit(input, vdot, dvdx, dv2dx, dv3dx, output);
  }

  ///
  /// \brief Compute the LHSBsplit for the implicit function
  /// \param[in] input A std::vector containing the field variables
  /// \param[in] vdot A std::vector containing the first-order time derivatives of the field variables
  /// \param[in] dvdx A std::vector containing the first-order space derivatives of the field variables
  /// \param[in] dv2dx A std::vector containing the second-order space derivatives of the field variables
  /// \param[in] dv3dx A std::vector containing the third-order space derivatives of the field variables
  /// \param[in,out] output A std::vector containing the computed residuals for the implicit function
  ///
  void LHSOpBsplit(std::vector<FloatingPointType> const& input,
    std::vector<FloatingPointType> const& vdot,
    std::vector<FloatingPointType> const& dvdx,
    std::optional<std::vector<FloatingPointType>> const& dv2dx,
    std::optional<std::vector<FloatingPointType>> const& dv3dx,
    std::vector<FloatingPointType>& output)
  {
    assert(m_boolLHSOpSet && "LHSOp function has not been set!");
    output = m_myLHSBsplit(input, vdot, dvdx, dv2dx, dv3dx, output);
  }

  ///
  /// \brief Jacobian LHS operator
  /// \param[in] input A std::vector containing the field variables
  /// \param[in] vdot A std::vector containing the first-order time derivatives of the field variables
  /// \param[in] dvdx A std::vector containing the first-order space derivatives of the field variables
  /// \param[in] dv2dx A std::vector containing the second-order space derivatives of the field variables
  /// \param[in] dv3dx A std::vector containing the third-order space derivatives of the field variables
  /// \param[in,out] output A std::vector containing the computed residuals for the implicit function
  /// \param[in] rowo
  /// \param[in] colo
  /// \param[in] derivo
  ///
  void JacLHSOpAsplit(std::vector<FloatingPointType> const& input,
    std::vector<FloatingPointType> const& vdot,
    std::vector<FloatingPointType> const& dvdx,
    std::optional<std::vector<FloatingPointType>> const& dv2dx,
    std::optional<std::vector<FloatingPointType>> const& dv3dx,
    std::vector<FloatingPointType>& output,
    int rowo,
    int colo,
    int derivo)
  {
    assert(m_boolJacLHSOpSet && "JacLHSOp function has not been set!");
    output = m_myJacLHSAsplit(input, vdot, dvdx, dv2dx, dv3dx, output, rowo, colo, derivo);
  }

  ///
  /// \brief Jacobian LHS operator
  /// \param[in] input A std::vector containing the field variables
  /// \param[in] vdot A std::vector containing the first-order time derivatives of the field variables
  /// \param[in] dvdx A std::vector containing the first-order space derivatives of the field variables
  /// \param[in] dv2dx A std::vector containing the second-order space derivatives of the field variables
  /// \param[in] dv3dx A std::vector containing the third-order space derivatives of the field variables
  /// \param[in,out] output A std::vector containing the computed residuals for the implicit function
  /// \param[in] rowo
  /// \param[in] colo
  /// \param[in] derivo
  ///
  void JacLHSOpBsplit(std::vector<FloatingPointType> const& input,
    std::vector<FloatingPointType> const& vdot,
    std::vector<FloatingPointType> const& dvdx,
    std::optional<std::vector<FloatingPointType>> const& dv2dx,
    std::optional<std::vector<FloatingPointType>> const& dv3dx,
    std::vector<FloatingPointType>& output,
    int rowo,
    int colo,
    int derivo)
  {
    assert(m_boolJacLHSOpSet && "JacLHSOp function has not been set!");
    output = m_myJacLHSBsplit(input, vdot, dvdx, dv2dx, dv3dx, output, rowo, colo, derivo);
  }

  ///
  /// \brief Setter for the RHS function
  /// \param[in] rhsFunc The callable object to set as the RHS function to
  ///
  void SetRHSFunc(SystemRHSFunctionType const& rhsFunc)
  {
    m_myRHS = rhsFunc;
    m_boolRHSSet = true;
  }

  ///
  /// \brief Setter for the RHS function
  ///
  virtual void SetRHSFunc() = 0;

  ///
  /// \brief Setter for the RHS Jacobian function
  /// \param[in] rhsJacobianFunc The callable object to set as the RHS Jacobian function
  ///
  void SetJacRHS(SystemRHSJacFunctionType const& rhsJacobianFunc)
  {
    m_myJacRHS = rhsJacobianFunc;
    m_boolJacRHSSet = true;
  }

  ///
  /// \brief Setter for the Jacobian RHS function
  ///
  virtual void SetJacRHS() = 0;

  ///
  /// \brief Setter for the LHS operator function. A/B split for implicit/explicit distinguishing
  /// \param[in] lhsa The callable object for the A split of the LHS function
  /// \param[in] lhsb The callable object for the B split of the LHS function
  ///
  void SetLHSOp(SystemLHSFunctionType const& lhsa, SystemLHSFunctionType const& lhsb)
  {
    m_myLHSAsplit = lhsa;
    m_myLHSBsplit = lhsb;

    m_boolLHSOpSet = true;
  }

  ///
  /// \brief Setter for the LHS operator function
  ///
  virtual void SetLHSOp() = 0;

  // Setter for the Jacobian LHS operator function.

  ///
  /// \brief Setter for the Jacobian LHS operator function. A/B split for implicit/explicit distinguishing
  /// \param[in] jaclhsa The callable object for the A split of the LHS Jacobian function
  /// \param[in] jaclhsb The callable object for the B split of the LHS Jacobian function
  ///
  void SetJacLHSOp(SystemLHSJacFunctionType const& jaclhsa, SystemLHSJacFunctionType const& jaclhsb)
  {
    m_myJacLHSAsplit = jaclhsa;
    m_myJacLHSBsplit = jaclhsb;
    m_boolJacLHSOpSet = true;
  }

  ///
  /// \brief Setter for the Jacobian LHS operator function
  ///
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
      auto const rhs = [](std::vector<FloatingPointType> const&,
                         std::vector<FloatingPointType>&,
                         int) -> std::vector<FloatingPointType> { return std::vector<FloatingPointType>(); };

      SetRHSFunc(rhs);
    }

    // If the Jacobian RHS function is not set, provide a default function.

    if (!m_boolJacRHSSet) {
      auto const jacRhs = [](std::vector<FloatingPointType> const&,
                            std::vector<FloatingPointType>&,
                            int) -> std::vector<FloatingPointType> { return std::vector<FloatingPointType>(); };

      SetJacRHS(jacRhs);
    }

    // If the LHS operator function is not set, provide a default function.

    if (!m_boolLHSOpSet) {
      auto const lhsa = [](std::vector<FloatingPointType> const&,
                          std::vector<FloatingPointType> const&,
                          std::vector<FloatingPointType> const&,
                          std::optional<std::vector<FloatingPointType>> const&,
                          std::optional<std::vector<FloatingPointType>> const&,
                          std::vector<FloatingPointType>&) -> std::vector<FloatingPointType>
      { return std::vector<FloatingPointType>(); };

      auto const lhsb = [](std::vector<FloatingPointType> const&,
                          std::vector<FloatingPointType> const&,
                          std::vector<FloatingPointType> const&,
                          std::optional<std::vector<FloatingPointType>> const&,
                          std::optional<std::vector<FloatingPointType>> const&,
                          std::vector<FloatingPointType>&) -> std::vector<FloatingPointType>
      { return std::vector<FloatingPointType>(); };

      SetLHSOp(lhsa, lhsb);
    }

    // If the Jacobian LHS operator function is not set, provide a default function.

    if (!m_boolJacLHSOpSet) {
      auto const jaclhsa = [](std::vector<FloatingPointType> const&,
                             std::vector<FloatingPointType> const&,
                             std::vector<FloatingPointType> const&,
                             std::optional<std::vector<FloatingPointType>> const&,
                             std::optional<std::vector<FloatingPointType>> const&,
                             std::vector<FloatingPointType>&,
                             int,
                             int,
                             int) { return std::vector<FloatingPointType>(); };

      auto const jaclhsb = [](std::vector<FloatingPointType> const&,
                             std::vector<FloatingPointType> const&,
                             std::vector<FloatingPointType> const&,
                             std::optional<std::vector<FloatingPointType>> const&,
                             std::optional<std::vector<FloatingPointType>> const&,
                             std::vector<FloatingPointType>&,
                             int,
                             int,
                             int) { return std::vector<FloatingPointType>(); };

      SetJacLHSOp(jaclhsa, jaclhsb);
    }
  }

  ///
  /// \brief Calls the private solution function.
  /// Takes a std::vector of inputs and then passes them to the solution function, with the result being written to a
  /// std::vector of outputs
  /// \param[in] input A std::vector containing the inputs
  /// \param[in,out] output A std::vector containing the ouputs
  ///
  void EvalSol(std::vector<FloatingPointType> const& input, std::vector<FloatingPointType>& output)
  {
    assert(m_solFound && "Solution function has not been set!");
    output = m_solution(input);
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

  // Note: pdetype and nd are declared only once.
  PDEType m_pdeType;
  PDEOptions m_myPdeOpts;
  PDEParams<PDEOptions> m_pdeParams;
  int m_nd {};
  int m_nv {};
  int m_nbound {};

  std::unordered_map<int, BoundaryCondition<FloatingPointType>> m_myBdry;
  const std::unordered_map<int, BoundaryCondition<FloatingPointType>> m_defaultBoundary;

private:
  SystemFunctionType m_groundFun;
  SystemFunctionType m_defaultInitconds;
  SystemFunctionType m_myInitConds;
  SystemLHSFunctionType m_myLHSAsplit;
  SystemLHSJacFunctionType m_myJacLHSAsplit;
  SystemLHSFunctionType m_myLHSBsplit;
  SystemLHSJacFunctionType m_myJacLHSBsplit;
  SystemRHSFunctionType m_myRHS;
  SystemRHSJacFunctionType m_myJacRHS;
  SystemFunctionType m_solution;
};


// ------------------------------------------------------------
// ManualPDESystem: a PDESystem extension where only
// - constructors, and
// - setter operations with user inputs
// are intended to be used.
//
// All "no-argument" virtual setup hooks from PDESystem are disabled
// and will throw an error if called.
// ------------------------------------------------------------
template<std::floating_point FloatingPointType, std::semiregular PDEOptions>
class ManualPDESystem final : public PDESystem<FloatingPointType, PDEOptions>
{
public:
  using Base = PDESystem<FloatingPointType, PDEOptions>;

  ManualPDESystem(PDEType pdeType, PDEParams<PDEOptions> const& params)
    : Base(pdeType, params)
  {}

  explicit ManualPDESystem(PDEParams<PDEOptions> const& params)
    : Base(PDEType::Unknown, params)
  {}

  ~ManualPDESystem() override = default;

  // ---------------------------
  // Disabled virtual hooks
  // ---------------------------

  void SetBottom() override { ThrowDisabled("SetBottom()"); }

  void SetInitCond() override { ThrowDisabled("SetInitCond()"); }

  auto InitCond() -> FloatingPointType override
  {
    ThrowDisabled("InitCond()");
    return FloatingPointType{};
  }

  void SetBdryCond() override { ThrowDisabled("SetBdryCond()"); }

  auto BdryCond() -> FloatingPointType override
  {
    ThrowDisabled("BdryCond()");
    return FloatingPointType{};
  }

  void SetRHSFunc() override { ThrowDisabled("SetRHSFunc()"); }

  void SetJacRHS() override { ThrowDisabled("SetJacRHS()"); }

  void SetLHSOp() override { ThrowDisabled("SetLHSOp()"); }

  void SetJacLHSOp() override { ThrowDisabled("SetJacLHSOp()"); }

private:
  [[noreturn]] static void ThrowDisabled(char const* fn)
  {
    throw std::logic_error(
      std::string("ManualPDESystem: call to '") + fn +
      "' is invalid.\n"
      "ManualPDESystem is configured only via constructors and explicit setters with user input, e.g.:\n"
      "  - SetBottom(seafloor_fun)\n"
      "  - SetInitCond(init_fun)\n"
      "  - SetBdryCond(bcs_map, numConds)\n"
      "  - SetRHSFunc(rhs_fun)\n"
      "  - SetJacRHS(jac_rhs_fun)\n"
      "  - SetLHSOp(lhsa, lhsb)\n"
      "  - SetJacLHSOp(jaclhsa, jaclhsb)\n"
    );
  }
};


} // namespace fcfd::pdemodel
