#pragma once

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <petsc.h>

#include "PetscOptions.hpp"
#include "PetscUtils.hpp"
#include "numerics/SolTools.hpp"

namespace fcfd::pdenumerics
{

/**
 * @class PetscWrap
 * @brief PETSc wrapper for solving PDE systems
 *
 * @details
 *
 * @tparam PDEOptions Type-specific PDE configuration options
 */
template<std::semiregular PDEOptions>
class PetscWrap : public SolTool<PetscReal, PDEOptions, PetscErrorCode, PetscOptions, PetscSolveOpts>
{
  using Base = SolTool<PetscReal, PDEOptions, PetscErrorCode, PetscOptions, PetscSolveOpts>;

public:
  /**
   * @brief Construct and initialize a new PETSc session
   *
   * @details Calls PetscInitialize() to set up the PETSc runtime environment
   * with the provided command-line arguments and help message.
   *
   * @param[in] argc The number of command-line arguments passed in to the program
   * @param[in] argv An array of command-line argument strings
   * @param[in] msg A help message describing the purpose of the program
   *
   * @throws PetscInitException if PetscInitialize fails
   */
  PetscWrap(int argc, char* argv[], const char* msg) noexcept(false)
    : m_session(argc, argv, msg)
  {
    if (IsPetscError(SetOptions())) [[unlikely]] {
      throw PetscInitException("Could not set custom PETSc options");
    }
  }

  /**
   * @brief Construct and initialize a new PETSc session
   *
   * @details Calls PetscInitialize() to set up the PETSc runtime environment
   * with the provided command-line arguments and help message.
   *
   * @param[in] argc The number of command-line arguments passed in to the program
   * @param[in] argv An array of command-line argument strings
   * @param[in] msg A help message describing the purpose of the program
   * @param[in] fields The unknown fields in the PDE system
   *
   * @throws PetscInitException if PetscInitialize fails
   */
  PetscWrap(int argc, char* argv[], const char* msg, std::vector<std::string> fields) noexcept(false)
    : PetscWrap(argc, argv, msg)
  {
    SetFields(std::move(fields));
    Base::SetInitFlag(0, FlagState::IsSet);
  }

  /**
   * @brief Construct and initialize a new PETSc session
   *
   * @details Calls PetscInitialize() to set up the PETSc runtime environment with the provided command-line arguments
   * and help message
   *
   * @param[in] argc The number of command-line arguments passed to the program
   * @param[in] argv An array of command-line argument strings
   * @param[in] msg A help message describing the purpose of the program
   * @param[in] tsType A string with the name of a PETSc TS method (the time/ODE integrators that
   * PETSc provides)
   * @param[in] fields The unknown fields in the PDE system
   *
   * @throws PetscInitException if PetscInitialize fails
   */
  PetscWrap(int argc, char* argv[], const char* msg, TSType const tsType, std::vector<std::string> fields)
    noexcept(false)
    : PetscWrap(argc, argv, msg, fields)
  {
    m_methodProps.tsType = tsType;
    Base::SetInitFlag(0, FlagState::IsSet);
  }

  /**
   * @brief Destructor
   */
  ~PetscWrap()
  {
    if (m_vec != nullptr) {
      VecDestroy(&m_vec);
    }
    if (m_myTS != nullptr) {
      TSDestroy(&m_myTS);
    }
    if (m_da != nullptr) {
      DMDestroy(&m_da);
    }
  }

  /**
   * @brief Write the names of the unknown fields in the PDE system to our DMDA instance
   *
   * @param[in] fieldNames The names of the unknown fields in the PDE system
   * @returns A PetscErrorCode instance on success or failure
   */
  auto SetFields(std::vector<std::string>&& fieldNames) -> PetscErrorCode
  {
    m_methodProps.fields = std::move(fieldNames);
    m_methodProps.numFields = std::size(m_methodProps.fields);
    return WriteFieldNamesToDM();
  }

  /**
   * @brief Set the solution options (an instance of PetscSolveOpts) to be used
   *
   * @returns PetscErrorCode indicating success or failure
   */
  auto InitSolMethod() -> PetscErrorCode override
  {
    return 0;
  }

  /**
   * @brief Set the PetscOptions for this current instance of PetscWrap
   * @param options The options to be set
   */
  void InitSolMethod(PetscOptions options)
  {
    m_methodProps = std::move(options);
    Base::SetInitFlag(0, FlagState::IsSet);
  }

  /**
   * @brief Creates a DMDA instance depending on the number of dimensions we're working with
   * @param[in] ndim The number of dimensions
   * @param[in] bdryConds The boundary type
   * @param[in] stendata Stencil data (i.e., global dimensions in x,y,z directions; degrees of freedom per node,
   * stencil width)
   * @returns A PetscErrorCode detailing success or failure
   */
  auto CreateGrid(int ndim, DMBoundaryType const bdryType, StencilData stendata) -> PetscErrorCode
  {
    assert(bdryType == DM_BOUNDARY_GHOSTED or bdryType == DM_BOUNDARY_PERIODIC and "Unsupported boundary type");
    assert(ndim >= 1 and ndim <= 3 and "Unsupported number of dimensions");
    assert(stendata.dof > 0 and "Degree of freedom must be a positive value");
    assert(stendata.stencilWidth >= 1 and "Stencil width must be greater than or equal to 1");

    Base::m_nd = ndim;

    if (ndim == 1) {
      Create1dStencilGrid(m_da, stendata, bdryType);
    }
    else if (ndim == 2) {
      Create2dStencilGrid(m_da, stendata, bdryType);
    }
    else if (ndim == 3) {
      Create3dStencilGrid(m_da, stendata, bdryType);
    }

    m_myStencilData = std::move(stendata);
    PetscCall(DMSetFromOptions(m_da));
    PetscCall(DMSetUp(m_da));

    Base::SetInitFlag(2, FlagState::IsSet);

    return 0;
  }

  /**
   * @brief Set the stencil grid weights to be used during computation
   *
   * @param[in] stengrid The stencil grid weights for first-order derivatives
   * @param[in] stengridd The stencil grid weights for second-order derivatives
   * @param[in] stengridd3 The stencil grid weights for third-order derivatives
   */
  void SetStencilGridWeights(std::vector<PetscReal> stengrid,
    std::optional<std::vector<PetscReal>> stengridd,
    std::optional<std::vector<PetscReal>> stengridd3)
  {
    assert(!stengrid.empty() and "This should always have values");

    m_myStencilGridd = std::move(stengrid);

    if (stengridd.has_value()) {
      m_myStencilGridd2 = std::move(stengridd);
    }

    if (stengridd3.has_value()) {
      m_myStencilGridd3 = std::move(stengridd3);
    }
  }

  /**
   * @brief Set the four local residual evaluation functions for use with the DMDA
   *
   * @return A PetscErrorCode indicating success or failure
   */
  auto MakeProbFuns() -> PetscErrorCode;

  /**
   * @brief Configure the TS solver with algorithm options
   *
   * @details Sets up the PETSc time stepping (TS_=) object with the current algorithm options stored in m_algOpts. This
   * includes configuring the TS type, initial time, maximum time, time step size, and final time handling behaviour.
   * Additionally, it processes any command-line options that override or supplement these settings.
   *
   * @returns PetscErrorCode indicating success (PETSC_SUCCESS) or failure
   * @pre m_myTS must be initialized (not nullptr)
   * @note This method is called automatically when SetSolOpts(PetscSolveOpts) is invoked
   * @see SetSolOpts(PetscSolveOpts)
   */
  auto SetSolOpts() -> PetscErrorCode override
  {
    assert(m_myTS != nullptr and "m_myTS is in an invalid state and must be initialized before use");

    PetscFunctionBeginUser;

    PetscCall(TSSetType(m_myTS, m_algOpts.tsType));
    PetscCall(TSSetTime(m_myTS, m_algOpts.initTime));
    PetscCall(TSSetMaxTime(m_myTS, m_algOpts.maxTime));
    PetscCall(TSSetTimeStep(m_myTS, m_algOpts.timeStep));
    PetscCall(TSSetExactFinalTime(m_myTS, m_algOpts.finalTimeOption));
    PetscCall(TSSetFromOptions(m_myTS));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /**
   * @brief Configure the TS solver with algorithm options
   *
   * @details Sets up the PETSc time-stepping object with the algorithm options stored in opts. This includes
   * configuring the TSType, initial time, maximum time, time-step size, and final time handling behaviour.
   * Additionally, it processes any command-line options that override or supplement these settings.
   *
   * @param[in] opts The algorithm options the TS solver should use
   *
   */
  void SetSolOpts(PetscSolveOpts opts) noexcept override
  {
    m_algOpts = std::move(opts);
    SetSolOpts();
  }

  /**
   * @brief Initialize the timestepper
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  auto InitializeTS() -> PetscErrorCode
  {
    assert(m_da and "The DM object is in an invalid state. CreateGrid() must be called before InitializeTS()");

    PetscFunctionBeginUser;

    PetscCall(TSCreate(PETSC_COMM_WORLD, &m_myTS));
    PetscCall(TSSetProblemType(m_myTS, TS_NONLINEAR));
    PetscCall(TSSetDM(m_myTS, m_da));
    PetscCall(TSSetApplicationContext(m_myTS, this));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /**
   * @brief Numerically solve the PDE
   *
   * @param[in] dmPtr The DM object
   * @param[in] vec A vector the same size as one obtained with DMCreateGlobalVector or DMCreateLocalVector
   * @returns PetscErrorCode indicating success or failure
   */
  auto NumericalSolve() -> PetscErrorCode override
  {
    assert(m_da and "The DM object is in an invalid state and must be initialized before use");
    assert(m_myTS and "The TS object is in an invalid state and must be initialized before use");

    PetscFunctionBeginUser;

    if (m_vec == nullptr) {
      PetscCall(DMCreateGlobalVector(m_da, &m_vec));
    }

    PetscCall(InitialState(m_da, m_vec, m_info));
    PetscCall(TSSolve(m_myTS, m_vec));

    if (m_callbackReport) {
      PetscCall(PrintCallbackReports());
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /**
   * @brief Calls a function on values of fields of a PDE and stores the evaluated result
   *
   * @details
   * This function evaluates the solution stored in the global vector at each local grid
   * point owned by the current MPI rank. The evaluation is performed by calling the
   * dimension-specific free functions EvaluateSolution1d/2d/3d, which iterate over the
   * DMDA local domain.
   *
   * The values returned by the user-provided function are stored in m_evalValues in the
   * same order as the grid traversal. The corresponding grid indices (i,j,k) are stored
   * in m_evalIJK. No assumptions are made about primitive variables; the stored values
   * correspond directly to the conservative degrees of freedom defined by the PDE.
   *
   * @param[in] interpol The function used to evaluate the fields of a PDE at a grid point
   * @returns PetscErrorCode indicating success or failure
   */
  auto EvaluateSolution(std::function<std::vector<double>(std::vector<PetscReal>&)> const& interpol)
    -> PetscErrorCode override
  {
    assert(m_da != nullptr && "DM object is not initialized");
    assert(m_vec != nullptr && "Solution vector is null. Call NumericalSolve() before evaluation");

    PetscFunctionBeginUser;

    // Clear previously stored evaluation results
    m_evalValues.clear();
    m_evalIJK.clear();

    // Precompute grid indices in standard DMDA local ordering so that indices
    // correspond exactly to the order in which the solution is evaluated
    if (m_info.dim == 1) {
      m_evalIJK.reserve(static_cast<std::size_t>(m_info.xm));
      for (PetscInt i = m_info.xs; i < m_info.xs + m_info.xm; ++i) {
        m_evalIJK.push_back({i, 0, 0});
      }
    }

    if (m_info.dim == 2) {
      m_evalIJK.reserve(static_cast<std::size_t>(m_info.xm) * static_cast<std::size_t>(m_info.ym));
      for (PetscInt j = m_info.ys; j < m_info.ys + m_info.ym; ++j) {
        for (PetscInt i = m_info.xs; i < m_info.xs + m_info.xm; ++i) {
          m_evalIJK.push_back({i, j, 0});
        }
      }
    }

    if (m_info.dim == 3) {
      m_evalIJK.reserve(static_cast<std::size_t>(m_info.xm) * static_cast<std::size_t>(m_info.ym)
        * static_cast<std::size_t>(m_info.zm));
      for (PetscInt k = m_info.zs; k < m_info.zs + m_info.zm; ++k) {
        for (PetscInt j = m_info.ys; j < m_info.ys + m_info.ym; ++j) {
          for (PetscInt i = m_info.xs; i < m_info.xs + m_info.xm; ++i) {
            m_evalIJK.push_back({i, j, k});
          }
        }
      }
    }

    // Wrapper around user function to store evaluated values while preserving
    // the required function signature for EvaluateSolution1d/2d/3d
    auto storeFunc = [this, &interpol](std::vector<PetscReal>& dofs) -> std::vector<PetscReal>
    {
      std::vector<double> outD = interpol(dofs);

      std::vector<PetscReal> out;
      out.reserve(outD.size());
      for (double v : outD) {
        out.push_back(static_cast<PetscReal>(v));
      }

      m_evalValues.push_back(out);
      return out;
    };

    PetscErrorCode ierr = PETSC_SUCCESS;

    // Dispatch to the dimension-specific evaluation routine
    if (m_info.dim == 1) {
      ierr = fcfd::pdenumerics::EvaluateSolution1d(m_da, m_vec, m_info, storeFunc);
    }

    if (m_info.dim == 2) {
      ierr = fcfd::pdenumerics::EvaluateSolution2d(m_da, m_vec, m_info, storeFunc);
    }

    if (m_info.dim == 3) {
      ierr = fcfd::pdenumerics::EvaluateSolution3d(m_da, m_vec, m_info, storeFunc);
    }

    PetscFunctionReturn(ierr);
  }

  PetscWrap(PetscWrap const&) = delete;
  auto operator=(PetscWrap const&) -> PetscWrap& = delete;
  PetscWrap(PetscWrap&&) noexcept = delete;
  auto operator=(PetscWrap&&) noexcept -> PetscWrap& = delete;

private:
  PetscSession m_session;

  PetscOptions m_methodProps {};
  PetscSolveOpts m_algOpts {};
  CallbackReports m_user {.iFuncCalled = PETSC_FALSE,
    .iJacobianCalled = PETSC_FALSE,
    .rhsFuncCalled = PETSC_FALSE,
    .rhsJacobianCalled = PETSC_FALSE};

  //! The time stepper we're using
  TS m_myTS {};
  DM m_da {};
  DMDALocalInfo m_info {};
  Vec m_vec {};
  PetscBool m_useRhsJacobian {PETSC_TRUE};
  PetscBool m_useIJacobian {PETSC_TRUE};
  PetscBool m_callbackReport {PETSC_TRUE};

  //! This stores our stencil data (i.e., the global dimensions in x, y, and z directions, degrees of freedom per node,
  //! stencil width)
  StencilData m_myStencilData {};

  std::vector<PetscReal> m_myStencilGridd {};
  std::optional<std::vector<PetscReal>> m_myStencilGridd2 {};
  std::optional<std::vector<PetscReal>> m_myStencilGridd3 {};

  //! Stores the last evaluation output per visited grid point (local portion on this rank)
  std::vector<std::vector<PetscReal>> m_evalValues {};

  //! Stores the corresponding (i,j,k) indices (local portion on this rank)
  std::vector<std::array<PetscInt, 3>> m_evalIJK {};

  /**
   * @brief Dynamically set user options at runtime
   *
   * @returns PetscErrorCode indicating success or failure
   */
  auto SetOptions() -> PetscErrorCode
  {
    PetscFunctionBeginUser;

    PetscOptionsBegin(PETSC_COMM_WORLD, "ptn_", "Options for patterns", "");

    PetscCall(PetscOptionsBool("-callback_report",
      "Report on which user-supplied callbacks were actually called",
      "",
      m_callbackReport,
      &m_callbackReport,
      nullptr));
    PetscCall(PetscOptionsBool(
      "-use_ijacobian", "Set callback DMDATSSetIJacobian()", "", m_useIJacobian, &m_useIJacobian, nullptr));
    PetscCall(PetscOptionsBool(
      "-use_rhsjacobian", "Set callback DMDATSSetRHSJacobian()", "", m_useRhsJacobian, &m_useRhsJacobian, nullptr));

    PetscOptionsEnd();

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  auto PrintCallbackReports() -> PetscErrorCode
  {
    PetscFunctionBeginUser;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Callback Report\nSolver Type: %s\n", m_algOpts.tsType));
    PetscCall(PetscPrintf(
      PETSC_COMM_WORLD, "  IFunction: %d  |  IJacobian:  %d\n", m_user.iFuncCalled, m_user.iJacobianCalled));
    PetscCall(PetscPrintf(
      PETSC_COMM_WORLD, "  RHSFunction: %d  |  RHSJacobian: %d\n", m_user.rhsFuncCalled, m_user.rhsJacobianCalled));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /**
   * @brief Write the names of fields to our DMDA instance
   *
   * @returns A PetscErrorCode indicating success or failure
   * @pre The DM object must be initialized and in a valid state
   */
  auto WriteFieldNamesToDM() -> PetscErrorCode;

  /**
   * @brief Set the four local residual evaluation functions for use with the DMDA
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  auto MakeProblemFunctions1d() -> PetscErrorCode;

  /**
   * @brief Set the four local residual evaluation functions for use with the DMDA
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  auto MakeProblemFunctions2d() -> PetscErrorCode;

  /**
   * @brief Set the four local residual evaluation functions for use with the DMDA
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  auto MakeProblemFunctions3d() -> PetscErrorCode;

  /**
   * @brief Set the initial state of the PDE system
   *
   * @param[in] distributedMesh The DM object
   * @param[in] vec The Vec object
   * @param[in] info The DMDALocalInfo object
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  auto InitialState(DM& distributedMesh, Vec& vec, DMDALocalInfo& info) -> PetscErrorCode;

  /**
   * @brief Computes the local right-hand side function for a 1D PDE system
   *
   * @param[in] info DMDALocalInfo structure containing grid dimensions and local indices for this
   * processor's domain
   * @param[in] time Current simulation time
   * @param[in] aY Array of flow variables at the current time step
   * @param[in] aG Array of computed time derivatives for flow variables
   * @param[in] pwp A void pointer to an instance of PetscWrap (the context)
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  static auto FormRHSFunctionLocal1d(DMDALocalInfo* info, PetscReal time, PetscReal* aY, PetscReal* aG, void* pwp)
    -> PetscErrorCode;

  /**
   * @brief Computes the local right-hand side function for a 2D PDE system
   *
   * @param[in] info DMDALocalInfo structure containing grid dimensions and local indices for this
   * processor's domain
   * @param[in] time Current simulation time
   * @param[in] aY Array of flow variables at the current time step
   * @param[in] aG Array of computed time derivatives for flow variables
   * @param[in] pwp A void pointer to an instance of PetscWrap (the context)
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  static auto FormRHSFunctionLocal2d(DMDALocalInfo* info, PetscReal time, PetscReal** aY, PetscReal** aG, void* pwp)
    -> PetscErrorCode;

  /**
   * @brief Computes the local right-hand side function for a 3D PDE system
   *
   * @param[in] info DMDALocalInfo structure containing grid dimensions and local indices for this
   * processor's domain
   * @param[in] time Current simulation time
   * @param[in] aY Array of flow variables at the current time step
   * @param[in] aG Array of computed time derivatives for flow variables
   * @param[in] pwp A void pointer to an instance of PetscWrap (the context)
   *
   * @returns A PetscErrorCode indicating success or failure
   */
  static auto FormRHSFunctionLocal3d(DMDALocalInfo* info, PetscReal time, PetscReal*** aY, PetscReal*** aG, void* pwp)
    -> PetscErrorCode;

  /**
   * @brief Computes the Jacobian of the right-hand side for a local DMDA domain partition of a 1D PDE system
   *
   * @details This function assembles the Jacobian matrix of the PDE system's right-hand side for a local
   * sub-domain managed by PETSc's Distributed Array (DMDA). It iterates over grid points, evaluates the Jacobian at
   * each point using the user-provided PetscWrap instance, and populates the PETSc matrix using stencil-based indexing.
   *
   * @param[in] info A pointer to DMDA local sub-domain information (grid indices and sizes)
   * @param[in] time Current simulation time
   * @param[in] aY An array containing the initial values of the unknown fields in the system
   * @param[in] asmMat The Jacobian matrix to be assembled (for implicit solvers)
   * @param[in] precondMat The material from which PETSc can build a preconditioner; typically the same as P
   * but may differ
   * @param[in] pwp A void pointer to an instance of PetscWrap (the context)
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormRHSJacobianLocal1d(
    DMDALocalInfo* info, PetscReal time, PetscReal* aY, Mat asmMat, Mat precondMat, void* pwp) -> PetscErrorCode;

  /**
   * @brief Computes the Jacobian of the right-hand side for a local DMDA domain partition of a 2D PDE system
   *
   * @details This function assembles the Jacobian matrix of the PDE system's right-hand side for a local
   * sub-domain managed by PETSc's Distributed Array (DMDA). It iterates over grid points, evaluates the Jacobian at
   * each point using the user-provided PetscWrap instance, and populates the PETSc matrix using stencil-based indexing.
   *
   * @param[in] info A pointer to DMDA local sub-domain information (grid indices and sizes)
   * @param[in] time Current simulation time
   * @param[in] aY An array containing the initial values of the unknown fields in the system
   * @param[in] asmMat The Jacobian matrix to be assembled (for implicit solvers)
   * @param[in] precondMat The material from which PETSc can build a preconditioner; typically the same as P
   * but may differ
   * @param[in] pwp A void pointer to an instance of PetscWrap (the context)
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormRHSJacobianLocal2d(
    DMDALocalInfo* info, PetscReal time, PetscReal** aY, Mat asmMat, Mat precondMat, void* pwp) -> PetscErrorCode;

  /**
   * @brief Computes the Jacobian of the right-hand side for a local DMDA domain partition of a 3D PDE system
   *
   * @details This function assembles the Jacobian matrix of the PDE system's right-hand side for a local
   * sub-domain managed by PETSc's Distributed Array (DMDA). It iterates over grid points, evaluates the Jacobian at
   * each point using the user-provided PetscWrap instance, and populates the PETSc matrix using stencil-based indexing.
   *
   * @param[in] info A pointer to DMDA local sub-domain information (grid indices and sizes)
   * @param[in] time Current simulation time
   * @param[in] aY An array containing the initial values of the unknown fields in the system
   * @param[in] asmMat The Jacobian matrix to be assembled (for implicit solvers)
   * @param[in] precondMat The material from which PETSc can build a preconditioner; typically the same as P
   * but may differ
   * @param[in] pwp A void pointer to an instance of PetscWrap (the context)
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormRHSJacobianLocal3d(
    DMDALocalInfo* info, PetscReal time, PetscReal*** aY, Mat asmMat, Mat precondMat, void* pwp) -> PetscErrorCode;

  // in system form  F(t,Y,dot Y) = G(t,Y),  compute F():
  //     F^u(t,a,q,a_t,q_t) = a_t +  Dx[q]
  //     F^v(t,a,q,a_t,q_t) = q_t +  Dx[q^2/a+ga^2/2] = q_t + 2q Dx[q]/a-q^2/a^2
  //     Dx[a]+g a Dx[a]
  // STARTIFUNCTION

  /**
   * @brief Compute the local implicit function residuals for a 1D PDE system
   *
   * @param[in] info   DMDA local information structure containing grid indices and dimensions
   * @param[in] time   Current time value
   * @param[in] aY     Array of field variables
   * @param[out] aYdot Array of time derivatives of field variables
   * @param[in,out] aF Array to store computed residuals for the implicit function
   * @param[in] pwp    A pointer to an instance of PetscWrap
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormIFunctionLocal1d(
    DMDALocalInfo* info, PetscReal time, PetscReal* aY, PetscReal* aYdot, PetscReal* aF, void* pwp) -> PetscErrorCode;

  /**
   * @brief Compute the local implicit function residuals for a 2D PDE system
   *
   * @param[in] info   DMDA local information structure containing grid indices and dimensions
   * @param[in] time   Current time value
   * @param[in] aY     Array of field variables
   * @param[out] aYdot Array of time derivatives of field variables
   * @param[in,out] aF Array to store computed residuals for the implicit function
   * @param[in] pwp    A pointer to an instance of PetscWrap
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormIFunctionLocal2d(
    DMDALocalInfo* info, PetscReal time, PetscReal** aY, PetscReal** aYdot, PetscReal** aF, void* pwp)
    -> PetscErrorCode;

  /**
   * @brief Compute the local implicit function residuals for a 3D PDE system
   *
   * @param[in] info   DMDA local information structure containing grid indices and dimensions
   * @param[in] time   Current time value
   * @param[in] aY     Array of field variables
   * @param[out] aYdot Array of time derivatives of field variables
   * @param[in,out] aF Array to store computed residuals for the implicit function
   * @param[in] pwp    A pointer to an instance of PetscWrap
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormIFunctionLocal3d(
    DMDALocalInfo* info, PetscReal time, PetscReal*** aY, PetscReal*** aYdot, PetscReal*** aF, void* pwp)
    -> PetscErrorCode;

  // in system form  F(t,Y,dot Y) = G(t,Y),  compute combined/shifted
  // Jacobian of F():
  //     J = (shift) dF/d(dot Y) + dF/dY
  // STARTIJACOBIAN

  /**
   * @brief Computes the Jacobian matrix for the implicit function of a 1D PDE system
   *
   * @param[in] info  A pointer to DMDA local sub-domain information (grid indices and sizes)
   * @param[in] time  Current simulation time
   * @param[in] aY    An array containing the initial values of the unknown fields in the system at
   * all grid points in the local sub-domain
   * @param[in] aYdot Array of time derivatives at all grid points
   * @param[in] shift Shift parameter (typically dt/theta) for implicit time integration; multiplies
   * the Ẏ contribution to the domain
   * @param[in] asmMat The Jacobian matrix to be assembled (for implicit solvers)
   * @param[in] precondMat The material from which PETSc can build a preconditioner matrix; typically the
   * same as P but may differ; filled with stencil values representing spatial discretization
   * @param[in] pwp   A void pointer to an instance of PetscWrap (the context)
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormIJacobianLocal1d(DMDALocalInfo* info,
    PetscReal time,
    PetscReal* aY,
    PetscReal* aYdot,
    PetscReal shift,
    Mat asmMat,
    Mat precondMat,
    void* pwp) -> PetscErrorCode;

  /**
   * @brief Computes the Jacobian matrix for the implicit function of a 2D PDE system
   *
   * @param[in] info  A pointer to DMDA local sub-domain information (grid indices and sizes)
   * @param[in] time  Current simulation time
   * @param[in] aY    An array containing the initial values of the unknown fields in the system at
   * all grid points in the local sub-domain
   * @param[in] aYdot Array of time derivatives at all grid points
   * @param[in] shift Shift parameter (typically dt/theta) for implicit time integration; multiplies
   * the Ẏ contribution to the domain
   * @param[in] asmMat The Jacobian matrix to be assembled (for implicit solvers)
   * @param[in] precondMat The material from which PETSc can build a preconditioner matrix; typically the
   * same as P but may differ; filled with stencil values representing spatial discretization
   * @param[in] pwp   A void pointer to an instance of PetscWrap (the context)
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormIJacobianLocal2d(DMDALocalInfo* info,
    PetscReal time,
    PetscReal** aY,
    PetscReal** aYdot,
    PetscReal shift,
    Mat asmMat,
    Mat precondMat,
    void* pwp) -> PetscErrorCode;

  /**
   * @brief Computes the Jacobian matrix for the implicit function of a 3D PDE system
   *
   * @param[in] info  A pointer to DMDA local sub-domain information (grid indices and sizes)
   * @param[in] time  Current simulation time
   * @param[in] aY    An array containing the initial values of the unknown fields in the system at
   * all grid points in the local sub-domain
   * @param[in] aYdot Array of time derivatives at all grid points
   * @param[in] shift Shift parameter (typically dt/theta) for implicit time integration; multiplies
   * the Ẏ contribution to the domain
   * @param[in] asmMat The Jacobian matrix to be assembled (for implicit solvers)
   * @param[in] precondMat The material from which PETSc can build a preconditioner matrix; typically the
   * same as P but may differ; filled with stencil values representing spatial discretization
   * @param[in] pwp   A void pointer to an instance of PetscWrap (the context)
   *
   * @returns PetscErrorCode indicating success or failure
   */
  static auto FormIJacobianLocal3d(DMDALocalInfo* info,
    PetscReal time,
    PetscReal*** aY,
    PetscReal*** aYdot,
    PetscReal shift,
    Mat asmMat,
    Mat precondMat,
    void* pwp) -> PetscErrorCode;
};

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::WriteFieldNamesToDM() -> PetscErrorCode
{
  assert(m_da and "The DM object is in an invalid state. CreateGrid() must be called before SetFields()");

  PetscFunctionBeginUser;

  for (PetscInt i = 0; auto const& field : m_methodProps.fields) {
    PetscCall(DMDASetFieldName(m_da, i, field.c_str()));
    ++i;
  }

  Base::SetInitFlag(0, FlagState::IsSet);

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::MakeProbFuns() -> PetscErrorCode
{
  assert(m_da != nullptr and "m_da is in an invalid state and must be initialized before use");

  PetscFunctionBeginUser;

  if (Base::m_nd == 1) {
    return MakeProblemFunctions1d();
  }
  else if (Base::m_nd == 2) {
    return MakeProblemFunctions2d();
  }
  else if (Base::m_nd == 3) {
    return MakeProblemFunctions3d();
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::MakeProblemFunctions1d() -> PetscErrorCode
{
  PetscFunctionBeginUser;

  PetscCall(DMDATSSetRHSFunctionLocal(m_da, INSERT_VALUES, (DMDATSRHSFunctionLocal)FormRHSFunctionLocal1d, this));
  PetscCall(DMDATSSetIFunctionLocal(m_da, INSERT_VALUES, (DMDATSIFunctionLocal)FormIFunctionLocal1d, this));

  if (m_useRhsJacobian) {
    PetscCall(DMDATSSetRHSJacobianLocal(m_da, (DMDATSRHSJacobianLocal)FormRHSJacobianLocal1d, this));
  }

  if (m_useIJacobian) {
    PetscCall(DMDATSSetIJacobianLocal(m_da, (DMDATSIJacobianLocal)FormIJacobianLocal1d, this));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::MakeProblemFunctions2d() -> PetscErrorCode
{
  PetscFunctionBeginUser;

  PetscCall(DMDATSSetRHSFunctionLocal(m_da, INSERT_VALUES, (DMDATSRHSFunctionLocal)FormRHSFunctionLocal2d, this));
  PetscCall(DMDATSSetIFunctionLocal(m_da, INSERT_VALUES, (DMDATSIFunctionLocal)FormIFunctionLocal2d, this));

  if (m_useRhsJacobian) {
    PetscCall(DMDATSSetRHSJacobianLocal(m_da, (DMDATSRHSJacobianLocal)FormRHSJacobianLocal2d, this));
  }

  if (m_useIJacobian) {
    PetscCall(DMDATSSetIJacobianLocal(m_da, (DMDATSIJacobianLocal)FormIJacobianLocal2d, this));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::MakeProblemFunctions3d() -> PetscErrorCode
{
  PetscFunctionBeginUser;

  PetscCall(DMDATSSetRHSFunctionLocal(m_da, INSERT_VALUES, (DMDATSRHSFunctionLocal)FormRHSFunctionLocal3d, this));
  PetscCall(DMDATSSetIFunctionLocal(m_da, INSERT_VALUES, (DMDATSIFunctionLocal)FormIFunctionLocal3d, this));

  if (m_useRhsJacobian) {
    PetscCall(DMDATSSetRHSJacobianLocal(m_da, (DMDATSRHSJacobianLocal)FormRHSJacobianLocal3d, this));
  }

  if (m_useIJacobian) {
    PetscCall(DMDATSSetIJacobianLocal(m_da, (DMDATSIJacobianLocal)FormIJacobianLocal3d, this));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::InitialState(DM& distributedMesh, Vec& vec, DMDALocalInfo& info) -> PetscErrorCode
{
  assert(Base::m_myPDE != nullptr && "PDE has not been set");

  PetscFunctionBeginUser;

  auto initFunc = [this](std::vector<PetscReal> const& coords) -> std::vector<PetscReal>
  { return this->Base::m_myPDE->InitCond(coords); };

  return SetInitialState(distributedMesh, vec, info, initFunc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// 1D /////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSFunctionLocal1d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal* aY, PetscReal* aG, void* pwp) -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 1,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  auto aYvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto aGvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
    std::copy_n(&aY[i * info->dof], info->dof, aYvec.begin());

    for (PetscInt field = 0; field < info->dof; ++field) {
      pwrap->Base::m_myPDE->RHS(std::as_const(aYvec), aGvec, field);
      std::copy_n(aGvec.begin(), info->dof, &aG[i * info->dof]);
    }
  }

  pwrap->m_user.rhsFuncCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSJacobianLocal1d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal* aY, Mat asmMat, Mat precondMat, void* pwp)
  -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 1,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  auto aYvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto outVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto colStenVec = std::vector<MatStencil>(static_cast<std::size_t>(info->dof), MatStencil {});
  auto rowSten = MatStencil {};

  for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
    std::copy_n(&aY[i * info->dof], info->dof, aYvec.begin());

    // Fill up colStenVec once before calling MatSetValuesStencil
    for (PetscInt field = 0; field < info->dof; ++field) {
      colStenVec[field] = {.k = 0, .j = 0, .i = i, .c = field};
    }

    for (PetscInt field = 0; field < info->dof; ++field) {
      rowSten = {.k = 0, .j = 0, .i = i, .c = field};
      pwrap->Base::m_myPDE->JacRHS(std::as_const(aYvec), outVec, field);

      PetscCall(
        MatSetValuesStencil(precondMat, 1, &rowSten, info->dof, colStenVec.data(), outVec.data(), INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(precondMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(precondMat, MAT_FINAL_ASSEMBLY));

  if (asmMat != precondMat) {
    PetscCall(MatAssemblyBegin(asmMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(asmMat, MAT_FINAL_ASSEMBLY));
  }

  pwrap->m_user.rhsJacobianCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormIFunctionLocal1d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal* aY, PetscReal* aYdot, PetscReal* aF, void* pwp)
  -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 1,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  // std::vectors for aY, aYdot, and aF

  auto ayVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ayDotVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto afASplitVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto afBSplitVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  // std::vectors for stencil grid weights

  auto const& myd1StenGrid = pwrap->m_myStencilGridd;
  auto const& myd2StenGrid = pwrap->m_myStencilGridd2;
  auto const& myd3StenGrid = pwrap->m_myStencilGridd3;

  // std::vectors for storing stencil grid contributions

  auto indx = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ind2x = std::invoke(
    [&info, &myd2StenGrid]() -> std::optional<std::vector<PetscReal>>
    {
      if (myd2StenGrid.has_value()) {
        return std::make_optional<std::vector<PetscReal>>(static_cast<std::size_t>(info->dof), 0);
      }

      return std::nullopt;
    });

  auto ind3x = std::invoke(
    [&info, &myd3StenGrid]() -> std::optional<std::vector<PetscReal>>
    {
      if (myd3StenGrid.has_value()) {
        return std::make_optional<std::vector<PetscReal>>(static_cast<std::size_t>(info->dof), 0);
      }

      return std::nullopt;
    });

  // We're only interested in the per-axis neighbours of the current grid point
  auto const stencilSize = (2 * info->sw) + 1;

  // Fill ghost cells at domain boundaries using nearest interior values

  if (info->bx == DM_BOUNDARY_GHOSTED) {
    FillGhostCells1d(*info, aY);
  }

  for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
    std::copy_n(&aY[i * info->dof], info->dof, ayVec.begin());
    std::copy_n(&aYdot[i * info->dof], info->dof, ayDotVec.begin());

    std::fill(indx.begin(), indx.end(), 0);

    for (PetscInt field = 0; field < info->dof; ++field) {
      for (PetscInt pos = 0; pos < stencilSize; ++pos) {
        // First derivative in x direction
        indx[field] += myd1StenGrid.at(pos) * aY[(i - info->sw + pos) * info->dof + field];
      }
    }

    if (myd2StenGrid.has_value()) {
      std::fill(ind2x->begin(), ind2x->end(), 0);

      for (PetscInt field = 0; field < info->dof; ++field) {
        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          // Second derivative in x direction
          ind2x->at(field) += myd2StenGrid->at(pos) * aY[(i - info->sw + pos) * info->dof + field];
        }
      }
    }

    if (myd3StenGrid.has_value()) {
      std::fill(ind3x->begin(), ind3x->end(), 0);

      for (PetscInt field = 0; field < info->dof; ++field) {
        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          // Third derivative in x direction
          ind3x->at(field) += myd3StenGrid->at(pos) * aY[(i - info->sw + pos) * info->dof + field];
        }
      }
    }

    // A-split: time term
    pwrap->Base::m_myPDE->LHSOpAsplit(ayVec,
      ayDotVec,
      indx,
      std::nullopt,
      std::nullopt,
      ind2x,
      std::nullopt,
      std::nullopt,
      ind3x,
      std::nullopt,
      std::nullopt,
      afASplitVec);

    // B-split: spatial flux term (uses indx = dvdx)
    pwrap->Base::m_myPDE->LHSOpBsplit(ayVec,
      ayDotVec,
      indx,
      std::nullopt,
      std::nullopt,
      ind2x,
      std::nullopt,
      std::nullopt,
      ind3x,
      std::nullopt,
      std::nullopt,
      afBSplitVec);

    for (PetscInt field = 0; field < info->dof; ++field) {
      aF[i * info->dof + field] = ayDotVec[field] + afBSplitVec[field];
    }
  }

  pwrap->m_user.iFuncCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormIJacobianLocal1d(DMDALocalInfo* info,
  [[maybe_unused]] PetscReal time,
  PetscReal* aY,
  PetscReal* aYdot,
  PetscReal shift,
  Mat asmMat,
  Mat precondMat,
  void* pwp) -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 1,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  // std::vectors for aY, and aYdot

  auto ayVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ayDotVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  // std::vectors for stencil grid weights

  auto const& myd1StenGrid = pwrap->m_myStencilGridd;
  auto const& myd2StenGrid = pwrap->m_myStencilGridd2;
  auto const& myd3StenGrid = pwrap->m_myStencilGridd3;

  // std::vectors for storing stencil grid contributions

  auto indx = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ind2x = std::invoke(
    [&info, &myd2StenGrid]() -> std::optional<std::vector<PetscReal>>
    {
      if (myd2StenGrid.has_value()) {
        return std::make_optional<std::vector<PetscReal>>(static_cast<std::size_t>(info->dof), 0);
      }

      return std::nullopt;
    });

  auto ind3x = std::invoke(
    [&info, &myd3StenGrid]() -> std::optional<std::vector<PetscReal>>
    {
      if (myd3StenGrid.has_value()) {
        return std::make_optional<std::vector<PetscReal>>(static_cast<std::size_t>(info->dof), 0);
      }

      return std::nullopt;
    });

  // We're only interested in the per-axis neighbours of the current grid point
  auto const stencilSize = (2 * info->sw) + 1;

  auto rowSten = MatStencil {};
  auto colStenVec = std::vector<MatStencil>(static_cast<std::size_t>(stencilSize), MatStencil {});

  auto vVec = std::vector<PetscReal>(static_cast<std::size_t>(stencilSize));

  auto vTmp = std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0);

  auto [vdASplitVec, vdBSplitVec] = std::make_pair(std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
    std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0));

  using PairOfOptionalVectors = std::pair<std::optional<std::vector<PetscReal>>, std::optional<std::vector<PetscReal>>>;

  auto [vd2ASplitVec, vd2BSplitVec] = std::invoke(
    [&stencilSize, &myd2StenGrid]() -> PairOfOptionalVectors
    {
      if (myd2StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  auto [vd3ASplitVec, vd3BSplitVec] = std::invoke(
    [&stencilSize, &myd3StenGrid]() -> PairOfOptionalVectors
    {
      if (myd3StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  if (info->bx == DM_BOUNDARY_GHOSTED) {
    FillGhostCells1d(*info, aY);
  }

  PetscCall(MatZeroEntries(precondMat));

  PetscAssert(info->st == DMDA_STENCIL_STAR,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONGSTATE,
    "This function implementation currently assumes the use of DMDA_STENCIL_STAR");

  // Iterate over interior points and assemble Jacobian entries

  for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
    std::copy_n(&aY[i * info->dof], info->dof, ayVec.begin());
    std::copy_n(&aYdot[i * info->dof], info->dof, ayDotVec.begin());

    // Compute first-order derivative stencil contributions
    std::fill(indx.begin(), indx.end(), 0);

    for (PetscInt field = 0; field < info->dof; ++field) {
      for (PetscInt pos = 0; pos < stencilSize; ++pos) {
        // First derivative in x-direction
        indx[field] += myd1StenGrid.at(pos) * aY[(i - info->sw + pos) * info->dof + field];
      }
    }

    // Compute second-order derivative stencil contributions if available
    if (myd2StenGrid.has_value()) {
      std::fill(ind2x->begin(), ind2x->end(), 0);

      for (PetscInt field = 0; field < info->dof; ++field) {
        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          // Second derivative in x direction
          ind2x->at(field) += myd2StenGrid->at(pos) * aY[(i - info->sw + pos) * info->dof + field];
        }
      }
    }

    if (myd3StenGrid.has_value()) {
      std::fill(ind3x->begin(), ind3x->end(), 0);

      for (PetscInt field = 0; field < info->dof; ++field) {
        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          // Third derivative in x direction
          ind3x->at(field) += myd3StenGrid->at(pos) * aY[(i - info->sw + pos) * info->dof + field];
        }
      }
    }

    // Loop over component row field components
    for (PetscInt compRow = 0; compRow < info->dof; ++compRow) {
      rowSten = {.k = 0, .j = 0, .i = i, .c = compRow};

      // Loop over component column field components
      for (PetscInt compCol = 0; compCol < info->dof; ++compCol) {
        // Compute off-diagonal Jacobian contributions from spatial stencil
        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          colStenVec[pos] = {.k = 0, .j = 0, .i = i - info->sw + pos, .c = compCol};
        }

        std::fill(vVec.begin(), vVec.end(), 0);

        // Add time derivative shift to diagonal
        if (compRow == compCol) {
          vVec[info->sw] += shift;
        }

        // Compute diagonal Jacobian contribution
        pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
          ayDotVec,
          indx,
          std::nullopt,
          std::nullopt,
          ind2x,
          std::nullopt,
          std::nullopt,
          ind3x,
          std::nullopt,
          std::nullopt,
          vdASplitVec,
          compRow,
          compCol,
          1);

        pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
          ayDotVec,
          indx,
          std::nullopt,
          std::nullopt,
          ind2x,
          std::nullopt,
          std::nullopt,
          ind3x,
          std::nullopt,
          std::nullopt,
          vdBSplitVec,
          compRow,
          compCol,
          1);

        // Compute off-diagonal Jacobian contributions from spatial stencil
        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          vVec[pos] += myd1StenGrid.at(pos) * (vdASplitVec[pos] + vdBSplitVec[pos]);
        }

        // Compute second-order derivative contributions if available
        if (myd2StenGrid.has_value()) {
          pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
            ayDotVec,
            indx,
            std::nullopt,
            std::nullopt,
            ind2x,
            std::nullopt,
            std::nullopt,
            ind3x,
            std::nullopt,
            std::nullopt,
            vd2ASplitVec.value(),
            compRow,
            compCol,
            2);

          pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
            ayDotVec,
            indx,
            std::nullopt,
            std::nullopt,
            ind2x,
            std::nullopt,
            std::nullopt,
            ind3x,
            std::nullopt,
            std::nullopt,
            vd2BSplitVec.value(),
            compRow,
            compCol,
            2);

          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            vVec[pos] += myd2StenGrid->at(pos) * (vd2ASplitVec->at(pos) + vd2BSplitVec->at(pos));
          }
        }

        // Compute third-order derivative contributions if available
        if (myd3StenGrid.has_value()) {
          pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
            ayDotVec,
            indx,
            std::nullopt,
            std::nullopt,
            ind2x,
            std::nullopt,
            std::nullopt,
            ind3x,
            std::nullopt,
            std::nullopt,
            vd3ASplitVec.value(),
            compRow,
            compCol,
            3);

          pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
            ayDotVec,
            indx,
            std::nullopt,
            std::nullopt,
            ind2x,
            std::nullopt,
            std::nullopt,
            ind3x,
            std::nullopt,
            std::nullopt,
            vd3BSplitVec.value(),
            compRow,
            compCol,
            3);

          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            vVec[pos] += myd3StenGrid->at(pos) * (vd3ASplitVec->at(pos) + vd3BSplitVec->at(pos));
          }
        }

        PetscCall(
          MatSetValuesStencil(precondMat, 1, &rowSten, stencilSize, colStenVec.data(), vVec.data(), INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(precondMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(precondMat, MAT_FINAL_ASSEMBLY));

  if (asmMat != precondMat) {
    PetscCall(MatAssemblyBegin(asmMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(asmMat, MAT_FINAL_ASSEMBLY));
  }

  pwrap->m_user.iJacobianCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// 2D /////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSFunctionLocal2d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal** aY, PetscReal** aG, void* pwp) -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 2,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  auto aYvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto aGvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      std::copy_n(&aY[j][i * info->dof], info->dof, aYvec.begin());

      for (PetscInt field = 0; field < info->dof; ++field) {
        pwrap->Base::m_myPDE->RHS(std::as_const(aYvec), aGvec, field);
      }

      std::copy_n(aGvec.begin(), info->dof, &aG[j][i * info->dof]);
    }
  }

  pwrap->m_user.rhsFuncCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSJacobianLocal2d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal** aY, Mat asmMat, Mat precondMat, void* pwp)
  -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 2,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  auto aYvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto outVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto colStenVec = std::vector<MatStencil>(static_cast<std::size_t>(info->dof), MatStencil {});
  auto rowSten = MatStencil {};

  for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      std::copy_n(&aY[j][i * info->dof], info->dof, aYvec.begin());

      for (PetscInt field = 0; field < info->dof; ++field) {
        pwrap->Base::m_myPDE->JacRHS(std::as_const(aYvec), outVec, field);
      }

      for (PetscInt field = 0; field < info->dof; ++field) {
        colStenVec[field] = {.k = 0, .j = j, .i = i, .c = field};
      }

      for (PetscInt field = 0; field < info->dof; ++field) {
        rowSten = {.k = 0, .j = j, .i = i, .c = field};

        PetscCall(
          MatSetValuesStencil(precondMat, 1, &rowSten, info->dof, colStenVec.data(), outVec.data(), INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(precondMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(precondMat, MAT_FINAL_ASSEMBLY));

  if (asmMat != precondMat) {
    PetscCall(MatAssemblyBegin(asmMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(asmMat, MAT_FINAL_ASSEMBLY));
  }

  pwrap->m_user.rhsJacobianCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormIFunctionLocal2d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal** aY, PetscReal** aYdot, PetscReal** aF, void* pwp)
  -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 2,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  // std::vectors for aY, aYdot, and aF

  auto ayVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ayDotVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto afASplitVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto afBSplitVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  // std::vectors for stencil grid weights

  auto const& myd1StenGrid = pwrap->m_myStencilGridd;
  auto const& myd2StenGrid = pwrap->m_myStencilGridd2;
  auto const& myd3StenGrid = pwrap->m_myStencilGridd3;

  // std::vectors for storing stencil grid contributions

  auto indx = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto indy = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  auto [ind2x, ind2y] = std::invoke(
    [&info, &myd2StenGrid]() -> std::pair<std::optional<std::vector<PetscReal>>, std::optional<std::vector<PetscReal>>>
    {
      if (myd2StenGrid.has_value()) {
        return std::make_pair(std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0));
      }

      return std::make_pair(std::nullopt, std::nullopt);
    });

  auto [ind3x, ind3y] = std::invoke(
    [&info, &myd3StenGrid]() -> std::pair<std::optional<std::vector<PetscReal>>, std::optional<std::vector<PetscReal>>>
    {
      if (myd3StenGrid.has_value()) {
        return std::make_pair(std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0));
      }

      return std::make_pair(std::nullopt, std::nullopt);
    });

  // We're only interested in the per-axis neighbours of the current grid point
  auto const stencilSize = (2 * info->sw) + 1;

  // Fill ghost cells at domain boundaries using nearest interior values
  if (info->bx == DM_BOUNDARY_GHOSTED and info->by == DM_BOUNDARY_GHOSTED) {
    FillGhostCells2d(*info, aY);
  }

  PetscAssert(info->st == DMDA_STENCIL_STAR,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONGSTATE,
    "This function implementation currently assumes the use of DMDA_STENCIL_STAR");

  for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      std::copy_n(&aY[j][i * info->dof], info->dof, ayVec.begin());
      std::copy_n(&aYdot[j][i * info->dof], info->dof, ayDotVec.begin());

      std::fill(indx.begin(), indx.end(), 0);
      std::fill(indy.begin(), indy.end(), 0);

      for (PetscInt pos = 0; pos < stencilSize; ++pos) {
        for (PetscInt field = 0; field < info->dof; ++field) {
          // First derivative in x direction
          indx[field] += myd1StenGrid.at(pos) * aY[j][(i - info->sw + pos) * info->dof + field];

          // First derivative in y direction
          indy[field] += myd1StenGrid.at(pos) * aY[j - info->sw + pos][i * info->dof + field];
        }
      }

      if (myd2StenGrid.has_value()) {
        std::fill(ind2x->begin(), ind2x->end(), 0);
        std::fill(ind2y->begin(), ind2y->end(), 0);

        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          for (PetscInt field = 0; field < info->dof; ++field) {
            // Second derivative in x direction
            ind2x->at(field) += myd2StenGrid->at(pos) * aY[j][(i - info->sw + pos) * info->dof + field];

            // Second derivative in y direction
            ind2y->at(field) += myd2StenGrid->at(pos) * aY[j - info->sw + pos][i * info->dof + field];
          }
        }
      }

      if (myd3StenGrid.has_value()) {
        std::fill(ind3x->begin(), ind3x->end(), 0);
        std::fill(ind3y->begin(), ind3y->end(), 0);

        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          for (PetscInt field = 0; field < info->dof; ++field) {
            // Third derivative in x direction
            ind3x->at(field) += myd3StenGrid->at(pos) * aY[j][(i - info->sw + pos) * info->dof + field];

            // Third derivative in y direction
            ind3y->at(field) += myd3StenGrid->at(pos) * aY[j - info->sw + pos][i * info->dof + field];
          }
        }
      }

      pwrap->Base::m_myPDE->LHSOpAsplit(
        ayVec, ayDotVec, indx, indy, std::nullopt, ind2x, ind2y, std::nullopt, ind3x, ind3y, std::nullopt, afASplitVec);

      pwrap->Base::m_myPDE->LHSOpBsplit(
        ayVec, ayDotVec, indx, indy, std::nullopt, ind2x, ind2y, std::nullopt, ind3x, ind3y, std::nullopt, afBSplitVec);

      for (PetscInt field = 0; field < info->dof; ++field) {
        aF[j][i * info->dof + field] = afASplitVec[field] + afBSplitVec[field];
      }
    }
  }

  pwrap->m_user.iFuncCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormIJacobianLocal2d(DMDALocalInfo* info,
  [[maybe_unused]] PetscReal time,
  PetscReal** aY,
  PetscReal** aYdot,
  PetscReal shift,
  Mat asmMat,
  Mat precondMat,
  void* pwp) -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 2,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  // std::vectors for aY, and aYdot

  auto ayVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ayDotVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  // std::vectors for stencil grid weights

  auto const& myd1StenGrid = pwrap->m_myStencilGridd;
  auto const& myd2StenGrid = pwrap->m_myStencilGridd2;
  auto const& myd3StenGrid = pwrap->m_myStencilGridd3;

  // std::vectors for storing stencil grid contributions

  auto [indx, indy] = std::make_pair(std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
    std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0));

  using PairOfOptionalVectors = std::pair<std::optional<std::vector<PetscReal>>, std::optional<std::vector<PetscReal>>>;

  auto [ind2x, ind2y] = std::invoke(
    [&info, &myd2StenGrid]() -> PairOfOptionalVectors
    {
      if (myd2StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  auto [ind3x, ind3y] = std::invoke(
    [&info, &myd3StenGrid]() -> PairOfOptionalVectors
    {
      if (myd3StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  // We're only interested in the per-axis neighbours of the current grid point
  auto const stencilSize = (2 * info->sw) + 1;

  auto rowSten = MatStencil {};
  auto colStenVec = std::vector<MatStencil>(static_cast<std::size_t>(stencilSize), MatStencil {});

  auto vVec = std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0);

  auto vTmp = std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0);

  auto [vdASplitVec, vdBSplitVec] = std::make_pair(std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
    std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0));

  auto [vd2ASplitVec, vd2BSplitVec] = std::invoke(
    [&stencilSize, &myd2StenGrid]() -> PairOfOptionalVectors
    {
      if (myd2StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  auto [vd3ASplitVec, vd3BSplitVec] = std::invoke(
    [&stencilSize, &myd3StenGrid]() -> PairOfOptionalVectors
    {
      if (myd3StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  if (info->bx == DM_BOUNDARY_GHOSTED and info->by == DM_BOUNDARY_GHOSTED) {
    FillGhostCells2d(*info, aY);
  }

  PetscAssert(info->st == DMDA_STENCIL_STAR,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONGSTATE,
    "This function implementation currently assumes the use of DMDA_STENCIL_STAR");

  PetscCall(MatZeroEntries(precondMat));

  // Assemble Jacobian for 2D domain
  for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      std::copy_n(&aY[j][i * info->dof], info->dof, ayVec.begin());
      std::copy_n(&aYdot[j][i * info->dof], info->dof, ayDotVec.begin());

      // Compute stencil contributions from spatial derivatives
      std::fill(indx.begin(), indx.end(), 0);
      std::fill(indy.begin(), indy.end(), 0);

      for (PetscInt pos = 0; pos < stencilSize; ++pos) {
        for (PetscInt field = 0; field < info->dof; ++field) {
          // First derivative in x-direction
          indx[field] += myd1StenGrid.at(pos) * aY[j][(i - info->sw + pos) * info->dof + field];

          // First derivative in y-direction
          indy[field] += myd1StenGrid.at(pos) * aY[j - info->sw + pos][i * info->dof + field];
        }
      }

      if (myd2StenGrid.has_value()) {
        std::fill(ind2x->begin(), ind2x->end(), 0);
        std::fill(ind2y->begin(), ind2y->end(), 0);

        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          for (PetscInt field = 0; field < info->dof; ++field) {
            // Second derivative in x-direction
            ind2x->at(field) += myd2StenGrid->at(pos) * aY[j][(i - info->sw + pos) * info->dof + field];

            // Second derivative in y-direction
            ind2y->at(field) += myd2StenGrid->at(pos) * aY[j - info->sw + pos][i * info->dof + field];
          }
        }
      }

      if (myd3StenGrid.has_value()) {
        std::fill(ind3x->begin(), ind3x->end(), 0);
        std::fill(ind3y->begin(), ind3y->end(), 0);

        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          for (PetscInt field = 0; field < info->dof; ++field) {
            // Third derivative in x direction
            ind3x->at(field) += myd3StenGrid->at(pos) * aY[j][(i - info->sw + pos) * info->dof + field];

            // Third derivative in y-direction
            ind3y->at(field) += myd3StenGrid->at(pos) * aY[j - info->sw + pos][i * info->dof + field];
          }
        }
      }

      // Assemble Jacobian entries for all field combinations
      for (PetscInt compRow = 0; compRow < info->dof; ++compRow) {
        rowSten = {.k = 0, .j = j, .i = i, .c = compRow};

        for (PetscInt compCol = 0; compCol < info->dof; ++compCol) {
          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            colStenVec[pos] = {.k = 0, .j = j - info->sw + pos, .i = i - info->sw + pos, .c = compCol};
          }

          std::fill(vVec.begin(), vVec.end(), 0.0);

          pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
            ayDotVec,
            indx,
            indy,
            std::nullopt,
            ind2x,
            ind2y,
            std::nullopt,
            ind3x,
            ind3y,
            std::nullopt,
            vTmp,
            compRow,
            compCol,
            0);

          if (compRow == compCol) {
            vVec[info->sw] += shift;
          }

          // Compute diagonal term (time derivative + LHS operator)
          pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
            ayDotVec,
            indx,
            indy,
            std::nullopt,
            ind2x,
            ind2y,
            std::nullopt,
            ind3x,
            ind3y,
            std::nullopt,
            vdASplitVec,
            compRow,
            compCol,
            1);

          pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
            ayDotVec,
            indx,
            indy,
            std::nullopt,
            ind2x,
            ind2y,
            std::nullopt,
            ind3x,
            ind3y,
            std::nullopt,
            vdBSplitVec,
            compRow,
            compCol,
            1);

          // Compute stencil contributions from first derivative
          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            vVec[pos] += myd1StenGrid.at(pos) * (vdASplitVec[pos] + vdBSplitVec[pos]);
          }

          // Compute second derivative contributions (if available)
          if (myd2StenGrid.has_value()) {
            pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
              ayDotVec,
              indx,
              indy,
              std::nullopt,
              ind2x,
              ind2y,
              std::nullopt,
              ind3x,
              ind3y,
              std::nullopt,
              vd2ASplitVec.value(),
              compRow,
              compCol,
              2);

            pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
              ayDotVec,
              indx,
              indy,
              std::nullopt,
              ind2x,
              ind2y,
              std::nullopt,
              ind3x,
              ind3y,
              std::nullopt,
              vd2BSplitVec.value(),
              compRow,
              compCol,
              2);

            // Compute off-diagonal Jacobian contributions from spatial stencil
            for (PetscInt pos = 0; pos < stencilSize; ++pos) {
              vVec[pos] += myd2StenGrid->at(pos) * (vd2ASplitVec->at(pos) + vd2BSplitVec->at(pos));
            }
          }

          // Compute third derivative contributions (if available)
          if (myd3StenGrid.has_value()) {
            pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
              ayDotVec,
              indx,
              indy,
              std::nullopt,
              ind2x,
              ind2y,
              std::nullopt,
              ind3x,
              ind3y,
              std::nullopt,
              vd3ASplitVec.value(),
              compRow,
              compCol,
              3);

            pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
              ayDotVec,
              indx,
              indy,
              std::nullopt,
              ind2x,
              ind2y,
              std::nullopt,
              ind3x,
              ind3y,
              std::nullopt,
              vd3BSplitVec.value(),
              compRow,
              compCol,
              3);

            // Compute off-diagonal Jacobian contributions from spatial stencil
            for (PetscInt pos = 0; pos < stencilSize; ++pos) {
              vVec[pos] += myd3StenGrid->at(pos) * (vd3ASplitVec->at(pos) + vd3BSplitVec->at(pos));
            }
          }

          // Insert row of Jacobian into matrix
          PetscCall(
            MatSetValuesStencil(precondMat, 1, &rowSten, stencilSize, colStenVec.data(), vVec.data(), INSERT_VALUES));
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(precondMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(precondMat, MAT_FINAL_ASSEMBLY));

  if (asmMat != precondMat) {
    PetscCall(MatAssemblyBegin(asmMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(asmMat, MAT_FINAL_ASSEMBLY));
  }

  pwrap->m_user.iJacobianCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// 3D /////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSFunctionLocal3d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal*** aY, PetscReal*** aG, void* pwp) -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 3,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  auto aYvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto aGvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
    for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
      for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
        std::copy_n(&aY[k][j][i * info->dof], info->dof, aYvec.begin());

        for (PetscInt field = 0; field < info->dof; ++field) {
          pwrap->Base::m_myPDE->RHS(std::as_const(aYvec), aGvec, field);
        }

        std::copy_n(aGvec.begin(), info->dof, &aG[k][j][i * info->dof]);
      }
    }
  }

  pwrap->m_user.rhsFuncCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSJacobianLocal3d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal*** aY, Mat asmMat, Mat precondMat, void* pwp)
  -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 3,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  auto aYvec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto outVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto colStenVec = std::vector<MatStencil>(static_cast<std::size_t>(info->dof), MatStencil {});
  auto rowSten = MatStencil {};

  for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
    for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
      for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
        std::copy_n(&aY[k][j][i * info->dof], info->dof, aYvec.begin());

        for (PetscInt field = 0; field < info->dof; ++field) {
          pwrap->Base::m_myPDE->JacRHS(std::as_const(aYvec), outVec, field);
        }

        for (PetscInt field = 0; field < info->dof; ++field) {
          colStenVec[field] = {.k = k, .j = j, .i = i, .c = field};
        }

        for (PetscInt field = 0; field < info->dof; ++field) {
          rowSten = {.k = k, .j = j, .i = i, .c = field};

          PetscCall(
            MatSetValuesStencil(precondMat, 1, &rowSten, info->dof, colStenVec.data(), outVec.data(), INSERT_VALUES));
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(precondMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(precondMat, MAT_FINAL_ASSEMBLY));

  if (asmMat != precondMat) {
    PetscCall(MatAssemblyBegin(asmMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(asmMat, MAT_FINAL_ASSEMBLY));
  }

  pwrap->m_user.rhsJacobianCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormIFunctionLocal3d(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal time, PetscReal*** aY, PetscReal*** aYdot, PetscReal*** aF, void* pwp)
  -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 3,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  // std::vectors for aY, aYdot, and aF

  auto ayVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ayDotVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto afASplitVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto afBSplitVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  // std::vectors for stencil grid weights

  auto const& myd1StenGrid = pwrap->m_myStencilGridd;
  auto const& myd2StenGrid = pwrap->m_myStencilGridd2;
  auto const& myd3StenGrid = pwrap->m_myStencilGridd3;

  // std::vectors for storing stencil grid contributions

  auto indx = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto indy = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto indz = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  using TripleOfVectors = std::tuple<std::optional<std::vector<PetscReal>>,
    std::optional<std::vector<PetscReal>>,
    std::optional<std::vector<PetscReal>>>;

  auto [ind2x, ind2y, ind2z] = std::invoke(
    [&info, &myd2StenGrid]() -> TripleOfVectors
    {
      if (myd2StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0)};
      }

      return {std::nullopt, std::nullopt, std::nullopt};
    });

  auto [ind3x, ind3y, ind3z] = std::invoke(
    [&info, &myd3StenGrid]() -> TripleOfVectors
    {
      if (myd3StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0)};
      }

      return {std::nullopt, std::nullopt, std::nullopt};
    });

  // We're only interested in the per-axis neighbours of the current grid point
  auto const stencilSize = (2 * info->sw) + 1;

  // Fill ghost cells at domain boundaries using nearest interior values
  if (info->bx == DM_BOUNDARY_GHOSTED and info->by == DM_BOUNDARY_GHOSTED and info->bz == DM_BOUNDARY_GHOSTED) {
    FillGhostCells3d(*info, aY);
  }

  PetscAssert(info->st == DMDA_STENCIL_STAR,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONGSTATE,
    "This function implementation currently assumes the use of DMDA_STENCIL_STAR");

  for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
    for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
      for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
        std::copy_n(&aY[k][j][i * info->dof], info->dof, ayVec.begin());
        std::copy_n(&aYdot[k][j][i * info->dof], info->dof, ayDotVec.begin());

        std::fill(indx.begin(), indx.end(), 0);
        std::fill(indy.begin(), indy.end(), 0);
        std::fill(indz.begin(), indz.end(), 0);

        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          for (PetscInt field = 0; field < info->dof; ++field) {
            // First derivative in x direction
            indx[field] += myd1StenGrid.at(pos) * aY[k][j][(i - info->sw + pos) * info->dof + field];

            // First derivative in y direction
            indy[field] += myd1StenGrid.at(pos) * aY[k][j - info->sw + pos][i * info->dof + field];

            // First derivative in z direction
            indz[field] += myd1StenGrid.at(pos) * aY[k - info->sw + pos][j][i * info->dof + field];
          }
        }

        if (myd2StenGrid.has_value()) {
          std::fill(ind2x->begin(), ind2x->end(), 0);
          std::fill(ind2y->begin(), ind2y->end(), 0);
          std::fill(ind2z->begin(), ind2z->end(), 0);

          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            for (PetscInt field = 0; field < info->dof; ++field) {
              // Second derivative in x direction
              ind2x->at(field) += myd2StenGrid->at(pos) * aY[k][j][(i - info->sw + pos) * info->dof + field];

              // Second derivative in y direction
              ind2y->at(field) += myd2StenGrid->at(pos) * aY[k][j - info->sw + pos][i * info->dof + field];

              // Second derivative in z direction
              ind2z->at(field) += myd2StenGrid->at(pos) * aY[k - info->sw + pos][j][i * info->dof + field];
            }
          }
        }

        if (myd3StenGrid.has_value()) {
          std::fill(ind3x->begin(), ind3x->end(), 0);
          std::fill(ind3y->begin(), ind3y->end(), 0);
          std::fill(ind3z->begin(), ind3z->end(), 0);

          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            for (PetscInt field = 0; field < info->dof; ++field) {
              // Third derivative in x direction
              ind3x->at(field) += myd3StenGrid->at(pos) * aY[k][j][(i - info->sw + pos) * info->dof + field];

              // Third derivative in y direction
              ind3y->at(field) += myd3StenGrid->at(pos) * aY[k][j - info->sw + pos][i * info->dof + field];

              // Third derivative in z direction
              ind3z->at(field) += myd3StenGrid->at(pos) * aY[k - info->sw + pos][j][i * info->dof + field];
            }
          }
        }

        pwrap->Base::m_myPDE->LHSOpAsplit(
          ayVec, ayDotVec, indx, indy, indz, ind2x, ind2y, ind2z, ind3x, ind3y, ind3z, afASplitVec);

        pwrap->Base::m_myPDE->LHSOpBsplit(
          ayVec, ayDotVec, indx, indy, indz, ind2x, ind2y, ind2z, ind3x, ind3y, ind3z, afBSplitVec);

        for (PetscInt field = 0; field < info->dof; ++field) {
          aF[k][j][i * info->dof + field] = afASplitVec[field] + afBSplitVec[field];
        }
      }
    }
  }

  pwrap->m_user.iFuncCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template<std::semiregular PDEOptions>
auto PetscWrap<PDEOptions>::FormIJacobianLocal3d(DMDALocalInfo* info,
  [[maybe_unused]] PetscReal time,
  PetscReal*** aY,
  PetscReal*** aYdot,
  PetscReal shift,
  Mat asmMat,
  Mat precondMat,
  void* pwp) -> PetscErrorCode
{
  PetscFunctionBeginUser;

  auto* const pwrap = static_cast<PetscWrap<PDEOptions>*>(pwp);

  PetscCheck(info and info->dim == 3,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONG,
    "Wrong number of dimensions or DMDALocalInfo has not been initialized");
  PetscCheck(pwrap->Base::m_myPDE != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The PDE system has not been set");
  PetscCheck(std::ssize(pwrap->m_methodProps.fields) == info->dof,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_INCOMP,
    "The number of fields and dof must be equal");

  // std::vectors for aY, and aYdot

  auto ayVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);
  auto ayDotVec = std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0);

  // std::vectors for stencil grid weights

  auto const& myd1StenGrid = pwrap->m_myStencilGridd;
  auto const& myd2StenGrid = pwrap->m_myStencilGridd2;
  auto const& myd3StenGrid = pwrap->m_myStencilGridd3;

  // std::vectors for storing stencil grid contributions

  auto [indx, indy, indz] = std::make_tuple(std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
    std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
    std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0));

  using TripleOfOptionalVectors = std::tuple<std::optional<std::vector<PetscReal>>,
    std::optional<std::vector<PetscReal>>,
    std::optional<std::vector<PetscReal>>>;

  auto [ind2x, ind2y, ind2z] = std::invoke(
    [&info, &myd2StenGrid]() -> TripleOfOptionalVectors
    {
      if (myd2StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0)};
      }

      return {std::nullopt, std::nullopt, std::nullopt};
    });

  auto [ind3x, ind3y, ind3z] = std::invoke(
    [&info, &myd3StenGrid]() -> TripleOfOptionalVectors
    {
      if (myd3StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(info->dof), 0)};
      }

      return {std::nullopt, std::nullopt, std::nullopt};
    });

  // We're only interested in the per-axis neighbours of the current grid point
  auto const stencilSize = (2 * info->sw) + 1;

  auto rowSten = MatStencil {};
  auto colStenVec = std::vector<MatStencil>(static_cast<std::size_t>(stencilSize), MatStencil {});

  auto vVec = std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0);

  auto vTmp = std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0);

  auto [vdASplit, vdBSplit] = std::make_pair(std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
    std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0));

  using PairOfOptionalVectors = std::pair<std::optional<std::vector<PetscReal>>, std::optional<std::vector<PetscReal>>>;

  auto [vd2ASplit, vd2BSplit] = std::invoke(
    [&stencilSize, &myd2StenGrid]() -> PairOfOptionalVectors
    {
      if (myd2StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  auto [vd3ASplit, vd3BSplit] = std::invoke(
    [&stencilSize, &myd3StenGrid]() -> PairOfOptionalVectors
    {
      if (myd3StenGrid.has_value()) {
        return {std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0),
          std::vector<PetscReal>(static_cast<std::size_t>(stencilSize), 0)};
      }

      return {std::nullopt, std::nullopt};
    });

  if (info->bx == DM_BOUNDARY_GHOSTED and info->by == DM_BOUNDARY_GHOSTED and info->bz == DM_BOUNDARY_GHOSTED) {
    FillGhostCells3d(*info, aY);
  }

  PetscAssert(info->st == DMDA_STENCIL_STAR,
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_WRONGSTATE,
    "This function implementation currently assumes the use of DMDA_STENCIL_STAR");

  PetscCall(MatZeroEntries(precondMat));

  // Assemble Jacobian for 3D domain
  for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
    for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
      for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
        std::copy_n(&aY[k][j][i * info->dof], info->dof, ayVec.begin());
        std::copy_n(&aYdot[k][j][i * info->dof], info->dof, ayDotVec.begin());

        // Compute stencil contributions from spatial derivatives
        std::fill(indx.begin(), indx.end(), 0);
        std::fill(indy.begin(), indy.end(), 0);
        std::fill(indz.begin(), indz.end(), 0);

        for (PetscInt pos = 0; pos < stencilSize; ++pos) {
          for (PetscInt field = 0; field < info->dof; ++field) {
            // First derivative in x-direction
            indx[field] += myd1StenGrid.at(pos) * aY[k][j][(i - info->sw + pos) * info->dof + field];

            // First derivative in y-direction
            indy[field] += myd1StenGrid.at(pos) * aY[k][j - info->sw + pos][i * info->dof + field];

            // First derivative in z-direction
            indz[field] += myd1StenGrid.at(pos) * aY[k - info->sw + pos][j][i * info->dof + field];
          }
        }

        if (myd2StenGrid.has_value()) {
          std::fill(ind2x->begin(), ind2x->end(), 0);
          std::fill(ind2y->begin(), ind2y->end(), 0);
          std::fill(ind2z->begin(), ind2z->end(), 0);

          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            for (PetscInt field = 0; field < info->dof; ++field) {
              // Second derivative in x-direction
              ind2x->at(field) += myd2StenGrid->at(pos) * aY[k][j][(i - info->sw + pos) * info->dof + field];

              // Second derivative in y-direction
              ind2y->at(field) += myd2StenGrid->at(pos) * aY[k][j - info->sw + pos][i * info->dof + field];

              // Second derivative in z-direction
              ind2z->at(field) += myd2StenGrid->at(pos) * aY[k - info->sw + pos][j][i * info->dof + field];
            }
          }
        }

        if (myd3StenGrid.has_value()) {
          std::fill(ind3x->begin(), ind3x->end(), 0);
          std::fill(ind3y->begin(), ind3y->end(), 0);
          std::fill(ind3z->begin(), ind3z->end(), 0);

          for (PetscInt pos = 0; pos < stencilSize; ++pos) {
            for (PetscInt field = 0; field < info->dof; ++field) {
              // Third derivative in x direction
              ind3x->at(field) += myd3StenGrid->at(pos) * aY[k][j][(i - info->sw + pos) * info->dof + field];

              // Third derivative in y direction
              ind3y->at(field) += myd3StenGrid->at(pos) * aY[k][j - info->sw + pos][i * info->dof + field];

              // Third derivative in z direction
              ind3z->at(field) += myd3StenGrid->at(pos) * aY[k - info->sw + pos][j][i * info->dof + field];
            }
          }
        }

        // Assemble Jacobian entries for all field contributions
        for (PetscInt compRow = 0; compRow < info->dof; ++compRow) {
          rowSten = {.k = k, .j = j, .i = i, .c = compRow};

          for (PetscInt compCol = 0; compCol < info->dof; ++compCol) {
            for (PetscInt pos = 0; pos < stencilSize; ++pos) {
              colStenVec[pos] = {
                .k = k - info->sw + pos, .j = j - info->sw + pos, .i = i - info->sw + pos, .c = compCol};
            }

            std::fill(vVec.begin(), vVec.end(), 0.0);

            pwrap->Base::m_myPDE->JacLHSOpBsplit(
              ayVec, ayDotVec, indx, indy, indz, ind2x, ind2y, ind2z, ind3x, ind3y, ind3z, vTmp, compRow, compCol, 0);

            if (compRow == compCol) {
              vVec[info->sw] += shift;
            }

            // Compute diagonal term (time derivative + LHS operator)
            pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
              ayDotVec,
              indx,
              indy,
              indz,
              ind2x,
              ind2y,
              ind2z,
              ind3x,
              ind3y,
              ind3z,
              vdASplit,
              compRow,
              compCol,
              1);

            pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
              ayDotVec,
              indx,
              indy,
              indz,
              ind2x,
              ind2y,
              ind2z,
              ind3x,
              ind3y,
              ind3z,
              vdBSplit,
              compRow,
              compCol,
              1);

            // Compute stencil contributions from first derivative
            for (PetscInt pos = 0; pos < stencilSize; ++pos) {
              vVec[pos] += myd1StenGrid.at(pos) * (vdASplit[pos] + vdBSplit[pos]);
            }

            // Compute second derivative contributions (if available)
            if (myd2StenGrid.has_value()) {
              pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
                ayDotVec,
                indx,
                indy,
                indz,
                ind2x,
                ind2y,
                ind2z,
                ind3x,
                ind3y,
                ind3z,
                vd2ASplit.value(),
                compRow,
                compCol,
                2);

              pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
                ayDotVec,
                indx,
                indy,
                indz,
                ind2x,
                ind2y,
                ind2z,
                ind3x,
                ind3y,
                ind3z,
                vd2BSplit.value(),
                compRow,
                compCol,
                2);

              // Compute off-diagonal Jacobian contributions from spatial stencil
              for (PetscInt pos = 0; pos < stencilSize; ++pos) {
                vVec[pos] += myd2StenGrid->at(pos) * (vd2ASplit->at(pos) + vd2BSplit->at(pos));
              }
            }

            // Compute third derivative contributions (if available)
            if (myd3StenGrid.has_value()) {
              pwrap->Base::m_myPDE->JacLHSOpAsplit(ayVec,
                ayDotVec,
                indx,
                indy,
                indz,
                ind2x,
                ind2y,
                ind2z,
                ind3x,
                ind3y,
                ind3z,
                vd3ASplit.value(),
                compRow,
                compCol,
                3);

              pwrap->Base::m_myPDE->JacLHSOpBsplit(ayVec,
                ayDotVec,
                indx,
                indy,
                indz,
                ind2x,
                ind2y,
                ind2z,
                ind3x,
                ind3y,
                ind3z,
                vd3BSplit.value(),
                compRow,
                compCol,
                3);

              // Compute off-diagonal Jacobian contributions from spatial stencil
              for (PetscInt pos = 0; pos < stencilSize; ++pos) {
                vVec[pos] += myd3StenGrid->at(pos) * (vd3ASplit->at(pos) + vd3BSplit->at(pos));
              }
            }

            PetscCall(
              MatSetValuesStencil(precondMat, 1, &rowSten, stencilSize, colStenVec.data(), vVec.data(), INSERT_VALUES));
          }
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(precondMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(precondMat, MAT_FINAL_ASSEMBLY));

  if (asmMat != precondMat) {
    PetscCall(MatAssemblyBegin(asmMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(asmMat, MAT_FINAL_ASSEMBLY));
  }

  pwrap->m_user.iJacobianCalled = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace fcfd::pdenumerics
