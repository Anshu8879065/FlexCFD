#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <petsc.h>

#include "PetscInitException.hpp"
#include "PetscOptions.hpp"
#include "SolTools.hpp"
#include "fcfd/itertools/SquareRange.hpp"

namespace fcfd::pdenumerics
{

///
/// \brief Check if a given PetscErrorCode is an error code
/// \param[in] code The code in question
/// \returns True if the code is an error code, and false otherwise
///
constexpr auto IsPetscError(PetscErrorCode code) noexcept -> bool
{
  return code != PETSC_SUCCESS;
}

struct CallBacks
{
  PetscBool iFuncCalled;
  PetscBool iJacobianCalled;
  PetscBool rhsFuncCalled;
  PetscBool rhsJacobianCalled;
};

template<typename PDEOptions>
class PetscWrap
  : public SolTool<PetscReal, PDEOptions, PetscErrorCode, PetscOptions, PetscSolveOpts>
{
public:
  ///
  /// \brief Default constructor
  ///
  explicit PetscWrap() noexcept = default;

  ///
  /// \brief Constructor. Creates a PetscWrap instance with `numOptions` number of options in the
  /// options database
  /// \param[in] numOptions The number of unknown fields in the PDE system
  ///
  explicit PetscWrap(int numFields) noexcept
    : PetscWrap()
  {
    m_methodProps.numFields = numFields;
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Constructor. Creates a PetscWrap instance with `optionNames` as the options in the
  /// options database
  /// \param[in] fieldNames The names of the unknown fields in the PDE system
  ///
  explicit PetscWrap(std::vector<std::string> fieldNames)
    : PetscWrap()
  {
    this->SetFields(fieldNames);
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Constructor
  ///
  /// Calls `PetscInitialize()` and sets RHS and I functions
  /// \param[in] argc The number of command-line arguments passed in to the program
  /// \param[in] argv A string containing a list of command-line arguments
  /// \param[in] msg A help message string describing the possible options to set to modify the
  /// program's behaviour
  /// \throws PetscInitException on failure of PetscInitialize or the setting of RHS and I functions
  ///
  PetscWrap(int argc, char** argv, const char* msg)
  {
    if (IsPetscError(PetscInitialize(&argc, &argv, nullptr, msg))) {
      throw PetscInitException();
    }

    if (IsPetscError(MakeProbFuns())) {
      throw PetscInitException();
    }
  }

  ///
  /// \brief Constructor
  /// \param[in] argc The number of command-line arguments passed in to the program
  /// \param[in] argv A string containing a list of the command-line arguments passed to the program
  /// \param[in] msg A help message string describing the possible options to set to modify the
  /// program's behaviour
  /// \param[in] numFields The number of unknown fields in the PDE system
  /// \throws PetscInitException on failure
  ///
  PetscWrap(int argc, char** argv, const char* msg, int numFields)
    : PetscWrap(argc, argv, msg)
  {
    m_methodProps.numFields = numFields;
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Constructor
  /// \param[in] argc The number of command-line arguments passed to the program
  /// \param[in] argv A string containing a list of the command-line arguments passed to the program
  /// \param[in] msg A help message string describing the possible options to set to modify the
  /// program's behaviour
  /// \param[in] tsType A string with the name of a PETSc TS method (the time/ODE integrators that
  /// PETSc provides)
  /// \param[in] numFields The number of unknown fields in the PDE system
  /// \throws PetscInitException on failure
  ///
  PetscWrap(int argc, char** argv, const char* msg, TSType tsType, int numFields)
    : PetscWrap(argc, argv, msg)
  {
    m_methodProps.numFields = numFields;
    m_methodProps.tsType = tsType;
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Constructor
  /// \param[in] argc The number of command-line arguments passed to the program
  /// \param[in] argv A string containing a list of the command-line arguments passed to the program
  /// \param[in] msg A help message string describing the possible options to set to modify the
  /// program's behaviour
  /// \param[in] tsType A string with the name of a PETSc TS method (the time/ODE integrators that
  /// PETSc provides)
  /// \param[in] fieldNames The names of the unknown fields in the PDE system
  /// \throws PetscInitException on failure
  ///
  PetscWrap(
    int argc, char** argv, const char* msg, TSType tsType, std::vector<std::string> fieldNames)
    : PetscWrap(argc, argv, msg)
  {
    m_methodProps.tsType = tsType;
    SetFields(fieldNames);
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Constructor
  /// \param[in] argc The number of command-line arguments passed to the program
  /// \param[in] argv A string containing a list of the command-line arguments passed to the program
  /// \param[in] msg A help message string describing the possible options to set to modify the
  /// program's behaviour
  /// \param[in] fieldNames The names of the unknown fields in the PDE system
  /// \throws PetscInitException on failure
  ///
  PetscWrap(int argc, char** argv, const char* msg, std::vector<std::string> fieldNames)
    : PetscWrap(argc, argv, msg)
  {
    SetFields(fieldNames);
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Copy constructor
  ///
  PetscWrap(PetscWrap const&) = default;

  ///
  /// \brief Copy-assignment operator
  ///
  auto operator=(PetscWrap const&) -> PetscWrap& = default;

  ///
  /// \brief Move constructor
  ///
  PetscWrap(PetscWrap&&) noexcept = default;

  ///
  /// \brief Move-assignment operator
  ///
  auto operator=(PetscWrap&&) noexcept -> PetscWrap& = default;

  ///
  /// \brief Destructor
  /// \details Calls PetscFinalize()
  ///
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
    if (IsPetscError(PetscFinalize())) {
      // throw PetscInitException();
    }
  }

  ///
  /// \brief
  ///
  virtual void SetFields()
  {
  }

  ///
  /// \brief Write the names of the unknown fields in the PDE system to our DMDA instance
  /// \param[in] fieldNames The names of the unknown fields in the PDE system
  /// \returns A PetscErrorCode instance on success or failure
  ///
  auto SetFields(std::vector<std::string> fieldNames) -> PetscErrorCode
  {
    m_methodProps.numFields = std::ssize(fieldNames);  // Possible narrowing conversion

    for (PetscInt i = 0; i < m_methodProps.numFields; ++i) {
      PetscCall(DMDASetFieldName(m_da, i, fieldNames[i].c_str()));
    }

    SetInitFlag(1, FlagState::IsSet);
    m_methodProps.fields = std::move(fieldNames);

    return PETSC_SUCCESS;
  }

  auto InitSolMethod() -> PetscErrorCode override;
  auto InitProblem() -> PetscErrorCode;

  ///
  /// \brief Creates a DMDA instance depending on the number of dimensions we're working with
  ///
  virtual void SetGrid()
  {
  }

  ///
  /// \brief Creates a DMDA instance depending on the number of dimensions we're working with
  /// \param[in] ndim The number of dimensions
  /// \param[in] bdryConds The boundary conditions
  /// \param[in] stengrid
  /// \param[in] stengridd
  /// \param[in] stengridd3
  /// \returns A PetscErrorCode detailing success or failure
  ///
  auto SetGrid(int ndim,
    std::vector<PetscInt> const& bdryConds,
    std::vector<PetscInt> stendata,
    std::vector<PetscReal>* stengrid,
    std::vector<PetscReal>* stengridd,
    std::vector<PetscReal>* stengridd3) -> PetscErrorCode
  {
    /*
     * We're only considering boundary conditions in the 1D, 2D, and 3D case, so this function works
     * on the assumption that this std::vector is of size 3
     */
    PetscAssert(bdryConds.size() == 3);

    m_nd = ndim;

    if (stengrid != nullptr) {
      m_myStencilGridd = std::move(*stengrid);
    }

    if (stengridd != nullptr) {
      m_myStencilGridd = std::move(*stengridd);
    }

    if (stengridd3 != nullptr) {
      m_myStencilGridd = std::move(*stengridd3);
    }

    m_inTmp = std::vector<PetscReal>(stendata[1], 0.0);
    m_outTmp = std::vector<PetscReal>(stendata[1], 0.0);

    switch (ndim) {
      // 1D
      case 1: {
        /*
         * Switch on the boundary conditions std::vector and adjust the boundary type in the
         * resulting DMDA accordingly. If the boundary condition is of value 2, we create a DMDA
         * with boundary type DM_BOUNDARY_GHOSTED. Otherwise if the boundary condition if of value
         * 3, we create a DMDA with boundary type DM_BOUNDARY_PERIODIC
         */
        switch (bdryConds[0]) {
          case 2: {
            PetscCall(DMDACreate1d(PETSC_COMM_WORLD,  // MPI communicator
              DM_BOUNDARY_GHOSTED,  // boundary type
              stendata[0],  // global dimension of the array (i.e., total number of grid points)
              stendata[1],  // number of degrees of freedom per node
              stendata[2],  // stencil width
              nullptr,  // array containing the number of nodes in the x direction (or NULL)
              &m_da));  // the resulting distributed array object
          } break;
          case 3: {
            PetscCall(DMDACreate1d(PETSC_COMM_WORLD,  // MPI communicator
              DM_BOUNDARY_PERIODIC,  // boundary type
              stendata[0],  // global dimension of the array (i.e., total number of grid points)
              stendata[1],  // number of degrees of freedom per node
              stendata[2],  // stencil width
              nullptr,  // array containing the number of nodes in the x direction (or NULL)
              &m_da));  // the resulting distributed array object
          } break;
        }
      } break;

      // 2D
      case 2: {
        /*
         * Switch on the boundary conditions std::vector and adjust the boundary type in the
         * resulting DMDA accordingly. If the boundary condition is of value 2, we create a DMDA
         * with boundary type DM_BOUNDARY_GHOSTED. Otherwise if the boundary condition if of value
         * 3, we create a DMDA with boundary type DM_BOUNDARY_PERIODIC
         */
        switch (bdryConds[1]) {
          case 2: {
            PetscCall(DMDACreate2d(PETSC_COMM_WORLD,  // MPI communicator
              DM_BOUNDARY_GHOSTED,  // The type of ghost nodes the x array has
              DM_BOUNDARY_GHOSTED,  // The type of ghost nodes the y array has
              DMDA_STENCIL_BOX,  // The stencil type
              stendata[0],  // The global dimension in the x direction of the array
              stendata[1],  // The global dimension in the y direction of the array
              PETSC_DECIDE,  // The corresponding number of processors in the x dimension (or
                             // PETSC_DECIDE to have it calculated)
              PETSC_DECIDE,  // The corresponding number of processors in the y dimension (or
                             // PETSC_DECIDE to have it calculated)
              stendata[2],  // The number of degrees of freedom per node
              stendata[3],  // The stencil width
              nullptr,  // The arrays containing the number of nodes in each cell along the x
                        // coordinate, or NULL
              nullptr,  // The arrays containing the number of nodes in each cell along the y
                        // coordinate, or NULL
              &m_da));  // The resulting distributed array object
          } break;
          case 3: {
            PetscCall(DMDACreate2d(PETSC_COMM_WORLD,  // MPI communicator
              DM_BOUNDARY_PERIODIC,  // the type of ghost nodes the x array has
              DM_BOUNDARY_PERIODIC,  // the type of ghost nodes the y array has
              DMDA_STENCIL_BOX,  // the stencil type
              stendata[0],  // the global dimension in the x direction of the array
              stendata[1],  // the global dimension in the y direction of the array
              PETSC_DECIDE,  // corresponding number of processors in the x dimension (or
                             // PETSC_DECIDE)
              PETSC_DECIDE,  // corresponding number of processors in the y dimension (or
                             // PETSC_DECIDE)
              stendata[2],  // the number of degrees of freedom per node
              stendata[3],  // the stencil width
              nullptr,  // arrays containing the number of nodes in each cell along the x
                        // coordinate
              nullptr,  // arrays containing the number of nodes in each cell along the y
                        // coordinate
              &m_da));  // the resulgint distributed array object
          } break;
        }
      }

      // 3D
      case 3: {
        /*
         * Switch on the boundary conditions std::vector and adjust the boundary type in the
         * resulting DMDA accordingly. If the boundary condition is of value 2, we create a DMDA
         * with boundary type DM_BOUNDARY_GHOSTED. Otherwise if the boundary condition if of value
         * 3, we create a DMDA with boundary type DM_BOUNDARY_PERIODIC
         */
        switch (bdryConds[2]) {
          case 2: {
            PetscCall(DMDACreate3d(PETSC_COMM_WORLD,  // MPI communicator
              DM_BOUNDARY_GHOSTED,  // type of ghost nodes the x array has
              DM_BOUNDARY_GHOSTED,  // type of ghost nodes the y array has
              DM_BOUNDARY_GHOSTED,  // type of ghost nodes the z array has
              DMDA_STENCIL_BOX,  // type of stencil
              stendata[0],  // global dimension in the x direction of the array
              stendata[1],  // global dimension in the y direction of the array
              stendata[2],  // global dimension in the z direction of the array
              PETSC_DECIDE,  // corresponding number of processors in x dimension (or PETSC_DECIDE)
              PETSC_DECIDE,  // corresponding number of processors in y dimension (or PETSC_DECIDE)
              PETSC_DECIDE,  // corresponding number of processors in z dimension (or PETSC_DECIDE)
              stendata[3],  // degrees of freedom per node
              stendata[4],  // stencil width
              nullptr,  // arrays containing number of nodes in each cell along x coordinate
              nullptr,  // arrays containing number of nodes in each cell along y coordinate
              nullptr,  // arrays containing number of nodes in each cell along z coordinate
              &m_da));  // resulting distributed array object
          } break;
          case 3: {
            PetscCall(DMDACreate3d(PETSC_COMM_WORLD,  // MPI communicator
              DM_BOUNDARY_PERIODIC,  // type of ghost nodes the x array has
              DM_BOUNDARY_PERIODIC,  // type of ghost nodes the y array has
              DM_BOUNDARY_PERIODIC,  // type of ghost nodes the z array has
              DMDA_STENCIL_BOX,  // type of stencil
              stendata[0],  // global dimension in the x direction of the array
              stendata[1],  // global dimension in the y direction of the array
              stendata[2],  // global dimension in the z direction of the array
              PETSC_DECIDE,  // corresponding number of processors in x dimension (or PETSC_DECIDE)
              PETSC_DECIDE,  // corresponding number of processors in y dimension (or PETSC_DECIDE)
              PETSC_DECIDE,  // corresponding number of processors in z dimension (or PETSC_DECIDE)
              stendata[3],  // degrees of freedom per node
              stendata[4],  // stencil width
              nullptr,  // arrays containing number of nodes in each cell along x coordinate
              nullptr,  // arrays containing number of nodes in each cell along y coordinate
              nullptr,  // arrays containing number of nodes in each cell along z coordinate
              &m_da));  // resulting distributed array object
          } break;
        }
        break;
      }

      default:
        break;
    }

    m_myStencilData = std::move(stendata);
    PetscCall(DMSetFromOptions(m_da));
    PetscCall(DMSetUp(m_da));

    SetInitFlag(2, FlagState::IsSet);

    return PETSC_SUCCESS;
  }

  auto MakeProbFuns() -> PetscErrorCode
  {
    PetscCall(DMDATSSetRHSFunctionLocal(
      m_da, INSERT_VALUES, (DMDATSRHSFunctionLocal)FormRHSFunctionLocal, this));
    PetscCall(DMDATSSetRHSJacobianLocal(m_da, (DMDATSRHSJacobianLocal)FormRHSJacobianLocal, this));
    PetscCall(
      DMDATSSetIFunctionLocal(m_da, INSERT_VALUES, (DMDATSIFunctionLocal)FormIFunctionLocal, this));
    PetscCall(DMDATSSetIJacobianLocal(m_da, (DMDATSIJacobianLocal)FormIJacobianLocal, this));

    return PETSC_SUCCESS;
  }

  auto SetSolOpts() -> PetscErrorCode override
  {
    if (GetInitFlag(3)) {
      PetscCall(TSSetType(m_myTS, m_algOpts.tsType));
      PetscCall(TSSetTime(m_myTS, m_algOpts.initTime));
      PetscCall(TSSetMaxTime(m_myTS, m_algOpts.maxTime));
      PetscCall(TSSetTimeStep(m_myTS, m_algOpts.timeStep));
      PetscCall(TSSetExactFinalTime(m_myTS, m_algOpts.finalTimeOption));
      PetscCall(TSSetFromOptions(m_myTS));
    }

    return PETSC_SUCCESS;
  }

  void SetSolOpts(const PetscSolveOpts& opts)
  {
    m_algOpts = opts;
    SetInitFlag(3, FlagState::IsSet);
    this->SetSolOpts();
  }

  ///
  /// \brief Set the TS scheme to be used
  /// \param[in] tsScheme The TS scheme to set
  ///
  void SetTSScheme(TS tsScheme)
  {
    m_myTS = tsScheme;
  }

  auto NumericalSolve() -> PetscErrorCode override
  {
    PetscCall(DMCreateGlobalVector(m_da, &m_vec));
    PetscCall(InitialState(m_da, m_vec));
    PetscCall(TSSolve(m_myTS, m_vec));

    return PETSC_SUCCESS;
  }

  virtual auto InterpolateSol(std::function<std::vector<double>(std::vector<PetscReal>)>& interpol)
    -> PetscErrorCode;
  // write implementation

private:
  using BaseClass = SolTool<PetscReal, PDEOptions, PetscErrorCode, PetscOptions, PetscSolveOpts>;
  using BaseClass::GetInitFlag;
  using BaseClass::m_nd;
  using BaseClass::SetInitFlag;

  PetscOptions m_methodProps;
  PetscSolveOpts m_algOpts;
  CallBacks m_user {.iFuncCalled = PETSC_FALSE,
    .iJacobianCalled = PETSC_FALSE,
    .rhsFuncCalled = PETSC_FALSE,
    .rhsJacobianCalled = PETSC_FALSE};

  TS m_myTS {};
  DM m_da {};
  DMDALocalInfo m_info {};
  TSType m_type {};
  Vec m_vec {};
  PetscBool m_noRhsJacobian {PETSC_FALSE};
  PetscBool m_noIJacobian {PETSC_FALSE};
  PetscBool m_callbackReport {PETSC_TRUE};

  std::vector<PetscInt> m_myStencilData;

  std::optional<std::vector<PetscReal>> m_myStencilGridd;
  std::optional<std::vector<PetscReal>> m_myStencilGriddd;
  std::optional<std::vector<PetscReal>> m_myStencilGridd3;

  std::vector<PetscReal> m_inTmp;
  std::vector<PetscReal> m_indTmp;
  std::optional<std::vector<PetscReal>> m_ind2Tmp;
  std::optional<std::vector<PetscReal>> m_ind3Tmp;

  std::vector<PetscReal> m_outTmp;
  std::vector<PetscReal> m_outdTmp;
  std::optional<std::vector<PetscReal>> m_outd2Tmp;
  std::optional<std::vector<PetscReal>> m_outd3Tmp;

  std::vector<PetscReal> m_vTmp;
  std::vector<PetscReal> m_vTmpTmp;
  std::vector<PetscReal> m_vdTmp;
  std::optional<std::vector<PetscReal>> m_vd2Tmp;
  std::optional<std::vector<PetscReal>> m_vd3Tmp;

  std::vector<MatStencil> m_colTmp;
  std::vector<MatStencil> m_coldTmp;
  std::optional<std::vector<MatStencil>> m_cold2Tmp;
  std::optional<std::vector<MatStencil>> m_cold3Tmp;

  //!
  MatStencil m_rowTmp {};
  MatStencil m_rowdTmp {};
  std::optional<MatStencil> m_rowd2Tmp;
  std::optional<MatStencil> m_rowd3Tmp;

  static auto FormRHSFunctionLocal(
    DMDALocalInfo* info, PetscReal t, PetscReal* aY, PetscReal* aG, void* pwp) -> PetscErrorCode;
  // ENDRHSFUNCTION

  static auto FormRHSJacobianLocal(
    DMDALocalInfo* info, PetscReal t, PetscReal* aY, Mat J, Mat P, void* pwp) -> PetscErrorCode;

  // in system form  F(t,Y,dot Y) = G(t,Y),  compute F():
  //     F^u(t,a,q,a_t,q_t) = a_t +  Dx[q]
  //     F^v(t,a,q,a_t,q_t) = q_t +  Dx[q^2/a+ga^2/2] = q_t + 2q Dx[q]/a-q^2/a^2
  //     Dx[a]+g a Dx[a]
  // STARTIFUNCTION
  static auto FormIFunctionLocal(
    DMDALocalInfo* info, PetscReal t, PetscReal* aY, PetscReal* aYdot, PetscReal* aF, void* pwp)
    -> PetscErrorCode;
  // ENDIFUNCTION

  // in system form  F(t,Y,dot Y) = G(t,Y),  compute combined/shifted
  // Jacobian of F():
  //     J = (shift) dF/d(dot Y) + dF/dY
  // STARTIJACOBIAN
  static auto FormIJacobianLocal(DMDALocalInfo* info,
    PetscReal t,
    PetscReal* aY,
    PetscReal* aYdot,
    PetscReal shift,
    Mat J,
    Mat P,
    void* pwp) -> PetscErrorCode;
};

///
/// \brief Computes the local right-hand side function for the shallow water equations
///
/// \param[in] info DMDALocalInfo structure containing grid dimensions and local indices for this
/// processor's domain
/// \param[in] time Current simulation time
/// \param[in] aY Array of flow variables at the current time step
/// \param[in] aG Array of computed time derivatives for flow variables
/// \param[in] pwp A void pointer to an instance of PetscWrap (the context)
/// \returns
///
template<typename PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSFunctionLocal(DMDALocalInfo* info,
  [[maybe_unused]] PetscReal time,
  PetscReal* aY,
  [[maybe_unused]] PetscReal* aG,
  void* pwp) -> PetscErrorCode
{
  auto* petscWrap = static_cast<PetscWrap<PDEOptions>*>(pwp);
  petscWrap->m_user.rhsFuncCalled = PETSC_TRUE;

  auto const& numDims = petscWrap->m_nd;
  auto const& numFields = std::ssize(petscWrap->m_methodProps.fields);
  auto const& inTmp = petscWrap->m_inTmp;
  auto const& vTmp = petscWrap->m_vTmp;

  itertools::SquareRange<PetscInt> xrange {info->xs, info->xs + info->xm};
  itertools::SquareRange<PetscInt> yrange {
    numDims >= 2 ? info->ys : 0, numDims >= 2 ? info->ys + info->ym : 0};
  itertools::SquareRange<PetscInt> zrange {
    numDims >= 3 ? info->zs : 0, numDims >= 3 ? info->zs + info->zm : 0};

  if (numDims == 1) {
    for (PetscInt i = info - xs; i < info->xs + info->xm; ++i) {
      for (PetscInt field = 0; field < numFields; ++field) {
        inTmp = std::vector(aY[i * numFields], aY[(i + 1) * numFields]);
        petscWrap->m_myPDE->RHS(inTmp, vTmp, field);
      }
    }
  }
  else if (numDims == 2) {
    PetscInt const ni = info->xs - info->xm;

    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        for (PetscInt field = 0; field < numFields; ++field) {
          inTmp = std::vector(aY[((i + ni) * j) * numFields], aY[((i + 1 + ni) * j) * numFields]);
          petscWrap->m_myPDE->RHS(inTmp, vTmp, field);
        }
      }
    }
  }
  else if (numDims == 3) {
    PetscInt const i = info->xm - info->xs;
    PetscInt const nj = info->ym - info->ys;

    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
          for (PetscInt field = 0; field < numFields; ++field) {
            inTmp = std::vector(aY[(i + ni * ((j + nj) * k)) * numFields],
              aY[(i + 1 + ni * (j + nj * k)) * numFields]);
            petscWrap->m_myPDE->RHS(inTmp, vTmp, field);
          }
        }
      }
    }
  }

  return PETSC_SUCCESS;
}

///
/// \brief Computes the Jacobian of the right-hand side for a local DMDA domain partition
///
/// This function assembles the Jacobian matrix of the PDE system's right-hand side for a local
/// sub-domain managed by PETSc's Distributed Array (DMDA). It iterates over grid points in 1D, 2D,
/// or 3D depending on problem dimensionality, evaluates the Jacobian at each point using the
/// user-provided PetscWrap instance, and populates the PETSc matrix using stencil-based indexing.
///
/// \param[in] info A pointer to DMDA local sub-domain information (grid indices and sizes)
/// \param[in] t Current simulation time
/// \param[in] aY An array containing the initial values of the unknown fields in the system
/// \param[in] J The Jacobian matrix to be assembled (for implicit solvers)
/// \param[in] P The material from which PETSc can build a preconditioner; typically the same as P
/// but may differ
/// \param[in] pwp A void pointer to an instance of PetscWrap (the context)
/// \returns PetscErrorCode indicating success or failure
///
template<typename PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSJacobianLocal(
  DMDALocalInfo* info, [[maybe_unused]] PetscReal t, PetscReal aY[], Mat J, Mat P, void* pwp)
  -> PetscErrorCode
{
  auto* petscWrap = static_cast<PetscWrap<PDEOptions>*>(pwp);
  petscWrap->m_user.rhsJacobianCalled = PETSC_TRUE;

  auto const& numDims = petscWrap->m_nd;

  using itertools::SquareRange;

  SquareRange<PetscInt> xrange {info->xs, info->xs + info->xm};
  SquareRange<PetscInt> yrange {
    numDims >= 2 ? info->ys : 0, numDims >= 2 ? info->ys + info->ym : 0};
  SquareRange<PetscInt> zrange {
    numDims >= 3 ? info->zs : 0, numDims >= 3 ? info->zs + info->zm : 0};

  auto const& numFields = std::ssize(petscWrap->m_methodProps.fields);
  auto& rowTmp = petscWrap->m_rowTmp;
  auto& colTmp = petscWrap->m_colTmp;
  auto& inTmp = petscWrap->m_inTmp;
  auto& vTmp = petscWrap->m_vTmp;

  if (numDims == 1) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      rowTmp.i = i;

      for (PetscInt field = 0; field < numFields; ++field) {
        colTmp[field].i = i;
        colTmp[field].c = field;

        rowTmp.c = field;
        inTmp = std::vector(aY[i * numFields], aY[(i + 1) * numFields]);
        petscWrap->m_myPDE->JacRHS(inTmp, vTmp, field);
        PetscCall(MatSetValuesStencil(P, 1, &rowTmp, 2, colTmp.data(), vTmp.data(), INSERT_VALUES));
      }
    }
  }
  else if (numDims == 2) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      rowTmp.i = i;
      PetscInt ni = info->xm - info->xs;

      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        rowTmp.j = j;

        for (PetscInt field = 0; field < numFields; ++field) {
          colTmp[field].i = i;
          colTmp[field].j = j;
          colTmp[field].c = field;

          rowTmp.c = field;
          inTmp = std::vector(aY[(i + ni * j) * numFields], aY[(i + 1 + ni * j) * numFields]);
          petscWrap->m_myPDE->JacRHS(inTmp, vTmp, field);
          PetscCall(
            MatSetValuesStencil(P, 1, &rowTmp, 2, colTmp.data(), vTmp.data(), INSERT_VALUES));
        }
      }
    }
  }
  else if (numDims == 3) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      PetscInt ni = info->xm - info->xs;
      rowTmp.i = i;

      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        rowTmp.j = j;
        PetscInt nj = info->ym - info->ys;

        for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
          rowTmp.k = k;

          for (PetscInt field = 0; field < numFields; ++field) {
            colTmp[field].i = i;
            colTmp[field].j = j;
            colTmp[field].k = k;
            colTmp[field].c = field;

            rowTmp.c = field;
            inTmp = std::vector(
              aY[(i + ni * (j + nj * k)) * numFields], aY[(i + 1 + ni * (j + nj * k)) * numFields]);
            petscWrap->m_myPDE->JacRHS(inTmp, vTmp, field);
            PetscCall(
              MatSetValuesStencil(P, 1, &rowTmp, 2, colTmp.data(), vTmp.data(), INSERT_VALUES));
          }
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  return PETSC_SUCCESS;
}

///
/// \brief FormIFunctionLocal - Compute the local implicit function residuals for the shallow water
/// equations
///
/// \param[in] info   DMDA local information structure containing grid indices and dimensions
/// \param[in] time   Current time value
/// \param[in] aY     Array of field variables
/// \param[out] aYdot Array of time derivatives of field variables
/// \param[in,out] aF Array to store computed residuals for the implicit function
/// \param[in] pwp    A pointer to an instance of PetscWrap
///
/// \returns PetscErrorCode indicating success (0) or error code
///
template<typename PDEOptions>
auto PetscWrap<PDEOptions>::FormIFunctionLocal(DMDALocalInfo* info,
  [[maybe_unused]] PetscReal time,
  PetscReal* aY,
  PetscReal* aYdot,
  PetscReal* aF,
  void* pwp) -> PetscErrorCode
{
  auto* petscWrap = static_cast<PetscWrap<PDEOptions>*>(pwp);
  petscWrap->m_user.iFuncCalled = PETSC_TRUE;

  auto numDims = petscWrap->m_nd;
  auto const& stencilData = petscWrap->m_myStencilData;
  auto const& numFields = std::ssize(petscWrap->m_methodProps.options);
  auto& inTmp = petscWrap->m_inTmp;
  auto& outTmp = petscWrap->m_outTmp;
  auto const& my1dStenGrid = petscWrap->m_myStencilGridd;
  auto const& my2dStenGrid = petscWrap->m_myStencilGriddd;
  auto& in1dTmp = petscWrap->m_indTmp;
  auto& in2dTmp = petscWrap->m_ind2Tmp;
  auto& in3dTmp = petscWrap->m_ind3Tmp;

  if (numDims == 1) {
    // Extract stencil width and compute half-width for ghost cell handling
    PetscInt const stencilWidth = stencilData[2];
    PetscInt const netWidth = std::floor(stencilWidth / 2.0);

    // Fill ghost cells at domain boundaries using nearest interior values

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        PetscInt const ghostLeft = info->xs - 1 - l;
        aY[(ghostLeft * numFields) + field] = aY[(info->xs * numFields) + field];

        PetscInt const ghostRight = info->xs + info->xm + l;
        aY[(ghostRight * numFields) + field] = aY[(info->xs + info->xm - 1) * numFields + field];
      }
    }

    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      inTmp = std::vector(aY[i * numFields], aY[(i + 1) * numFields]);
      outTmp = std::vector(aF[i * numFields], aF[(i + 1) * numFields]);

      if (my1dStenGrid.has_value()) {
        in1dTmp = std::vector<PetscReal>(numFields, 0.0f);

        for (PetscInt l = 0; l < stencilWidth; ++l) {
          in1dTmp[l] += my1dStenGrid[l] * aY[(i - netWidth + l) * numFields];
        }
      }

      if (my2dStenGrid.has_value()) {
        in2dTmp = std::vector<PetscReal>(numFields, 0.0f);

        for (PetscInt l = 0; l < stencilWidth; ++l) {
          in2dTmp += my2dStenGrid[l] * aY[(i - netWidth + l) * numFields];
        }
      }

      petscWrap->m_myPDE->LHSOpAsplit(inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, outTmp);
    }
  }
  else if (numDims == 2) {
    auto const& vec = petscWrap->m_vec;
    PetscReal*** ptr = nullptr;
    DMDAVecGetArrayDOF(info->da, vec, &ptr);

    // Extract stencil width and compute half-width for ghost cell handling
    PetscInt const stencilWidth = stencilData[2];
    PetscInt const netWidth = std::floor(stencilWidth / 2.0);

    // Fill ghost cells at domain boundaries using nearest interior values

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        // Left and right boundaries (loop over y)
        for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
          ptr[j][info->xs - 1 - l][field] = ptr[j][info->xs][f];  // left
          ptr[j][info->xs + info->xm + l][field] = ptr[j][info->xs + info->xm - 1][field];
        }

        // Bottom and top boundaries (loop over x)
        for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
          ptr[info->ys - 1 - l][i][field] = ptr[info->ys][i][field];
          ptr[info->ys + info->ym + l][i][field] = ptr[info->ys + info->ym - 1][i][field];
        }
      }
    }

    DMDAVecRestoreArrayDOF(info->da, vec, &ptr);

    PetscInt const ni = info->xm - info->xs;

    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        inTmp = std::vector(aY[(i + ni * j) * numFields], aY[(i + 1 + ni * j) * numFields]);
        outTmp = std::vector(aF[(i + ni * j) * numFields], aF[(i + 1 + ni * j) * numFields]);

        if (my1dStenGrid.has_value()) {
          in1dTmp = std::vector<PetscReal>(numFields, 0.0f);

          for (PetscInt l = 0; l < stencilWidth; ++l) {
            in1dTmp[l] += my1dStenGrid[l] * aY[(((i + ni) * j) - netWidth + l) * numFields];
          }
        }

        if (my2dStenGrid.has_value()) {
          in2dTmp = std::vector<PetscReal>(numFields, 0.0f);

          for (PetscInt l = 0; l < stencilWidth; ++l) {
            in2dTmp[l] += my2dStenGrid[l] * aY[(((i + ni) * j) - netWidth + l) * numFields];
          }
        }

        petscWrap->m_myPDE->JacLHSOpAsplit(inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, outTmp);
      }
    }
  }
  else if (numDims == 3) {
    PetscInt const stencilWidth = stencilData[2];
    PetscInt const netWidth = std::floor(stencilWidth / 2.0);

    auto const& vec = petscWrap->m_vec;
    PetscReal**** ptr = nullptr;
    DMDAVecGetArrayDOF(info->da, vec, ptr);

    // Fill ghost cells at domain boundaries using nearest interior values

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        // Face 1 and 2: x boundaries (left/right)
        // Loop over y and z

        for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
          for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
            ptr[k][j][info->xs - 1 - l][field] = ptr[k][j][info->xs][field];
            ptr[k][j][info->xs + info->xm + l][field] = ptr[k][j][info->xs + info->xm - 1][field];
          }
        }

        // Face 3 and 4: y boundaries (top/bottom)
        // Loop over x and z

        for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
          for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
            ptr[k][info->ys - 1 - l][i][field] = ptr[k][info->ys][i][field];
            ptr[k][info->ys + info->ym + l][i][field] = ptr[k][info->ys + info->ym - 1][i][field];
          }
        }

        // Face 5 and 6: z boundaries (front/back)
        // Loop over x and y

        for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
          for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
            ptr[info->zs - 1 - l][j][i][field] = ptr[info->zs][j][i][field];
            ptr[info->z + info->zm + l][j][i][field] = ptr[info->zs + info->zm - 1][j][i][field];
          }
        }
      }
    }

    DMDAVecRestoreArrayDOF(info->da, vec, &ptr);

    PetscInt const ni = info->xm - info->xs;
    PetscInt const nj = info->ym - info->ys;

    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
          inTmp = std::vector(
            aY[(i + ni * (j + nj + k)) * numFields], aY[(i + 1 + ni * (j + nj * k)) * numFields]);
          outTmp = std::vector(
            aF[(i + ni * (j + nj + k)) * numFields], aF[(i + 1 + ni * (j + nj * k)) * numFields]);

          if (my1dStenGrid.has_value()) {
            in1dTmp = std::vector(numFields, 0.0f);

            for (PetscInt l = 0; l < stencilWidth; ++l) {
              in1dTmp[l] +=
                my1dStenGrid[l] * aY[(i + ni * (j + nj + k) - netWidth + l) * numFields];
            }
          }

          if (my2dStenGrid.has_value()) {
            in2dTmp = std::vector(numFields, 0.0f);

            for (PetscInt l = 0; l < stencilWidth; ++l) {
              in2dTmp[l] +=
                my2dStenGrid[l] * aY[(i + ni * (j + nj + k) - netWidth + l) * numFields];
            }
          }

          petscWrap->m_myPDE->JacLHSOpAsplit(inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, outTmp);
        }
      }
    }
  }

  return PETSC_SUCCESS;
}

///
/// \brief Computes the Jacobian matrix for the implicit function F(t, Y, Ẏ) in the Shallow Water
/// equations
///
/// \param[in] info  A pointer to DMDA local sub-domain information (grid indices and sizes)
/// \param[in] time  Current simulation time
/// \param[in] aY    An array containing the initial values of the unknown fields in the system at
/// all grid points in the local sub-domain
/// \param[in] aYdot Array of time derivatives at all grid points
/// \param[in] shift Shift parameter (typically dt/theta) for implicit time integration; multiplies
/// the Ẏ contribution to the domain
/// \param[in,out] J The Jacobian matrix to be assembled (for implicit solvers)
/// \param[in,out] P The material from which PETSc can build a preconditioner matrix; typically the
/// same as P but may differ; filled with stencil values representing spatial discretization
/// \param[in] pwp   A void pointer to an instance of PetscWrap (the context)
/// \returns PetscErrorCode indicating success or failure
///
template<typename PDEOptions>
auto PetscWrap<PDEOptions>::FormIJacobianLocal(DMDALocalInfo* info,
  [[maybe_unused]] PetscReal time,
  PetscReal* aY,
  [[maybe_unused]] PetscReal* aYdot,
  PetscReal shift,
  Mat J,
  Mat P,
  void* pwp) -> PetscErrorCode
{
  PetscCall(MatZeroEntries(P));

  auto* petscWrap = static_cast<PetscWrap<PDEOptions>*>(pwp);
  petscWrap->m_user.iJacobianCalled = PETSC_TRUE;

  auto const& numDims = petscWrap->m_nd;
  auto const& numFields = std::ssize(petscWrap->m_methodProps.options);
  auto const& stenData = petscWrap->m_myStencilData;
  auto& rowTmp = petscWrap->m_rowTmp;
  auto& inTmp = petscWrap->m_inTmp;
  auto const& my1dStenGrid = petscWrap->m_myStencilGridd;
  auto const& my2dStenGrid = petscWrap->m_myStencilGriddd;
  auto& in1dTmp = petscWrap->m_indTmp;
  auto& in2dTmp = petscWrap->m_ind2Tmp;
  auto& in3dTmp = petscWrap->m_ind3Tmp;
  auto& vTmp = petscWrap->m_vTmp;
  auto& vTmpTmp = petscWrap->m_vTmpTmp;
  auto& colTmp = petscWrap->m_colTmp;

  if (numDims == 1) {
    // Extract stencil width and compute half-width for ghost cell handling
    PetscInt const stencilWidth = stenData[2];
    PetscInt const netWidth = std::floor(stencilWidth / 2.0);

    // Fill ghost cells at domain boundaries using nearest interior values
    for (PetscInt curr = 0; curr < netWidth; ++curr) {
      for (PetscInt field = 0; field < numFields; ++field) {
        // Ghost cells at left boundary
        PetscInt const left = info->xs - 1 - curr;
        aY[left + field] = aY[left + field + numFields];

        // Ghost cells at right boundary
        PetscInt const right = info->xs + info->xm + field;
        aY[(right - 1) * numFields + field] = aY[(right - 2) * numFields + field];
      }
    }

    // Iterate over interior points and assemble Jacobian entries

    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      inTmp[i] = aY[i * numFields];

      // Compute first derivative stencil contributions if available
      if (my1dStenGrid.has_value()) {
        in1dTmp = std::vector<PetscReal>(numFields, 0.0f);

        for (PetscInt l = 0; l < stencilWidth; ++l) {
          for (PetscInt field = 0; field < numFields; ++field) {
            in1dTmp[field] += my1dStenGrid[l] * aY[numFields * (i - netWidth + l)];
          }
        }
      }

      // Set row index for current point
      rowTmp.i = i;

      // Loop over component row field components
      for (PetscInt compRow = 0; compRow < numFields; ++compRow) {
        rowTmp.c = compRow;

        // Loop over component column field components
        for (PetscInt compCol = 0; compCol < numFields; ++compCol) {
          // Compute diagonal Jacobian contribution
          if (my1dStenGrid.has_value()) {
            petscWrap->m_myPDE->JacLHSOpAsplit(
              inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmp, compRow, compCol, 0);

            // Add time derivative shift to diagonal
            if (compRow == compCol) {
              vTmp[netWidth] += shift;
            }

            // Compute off-diagonal Jacobian contributions from spatial stencil
            for (PetscInt l = 0; l < stencilWidth; ++l) {
              colTmp[l].c = compCol;
              colTmp[l].i = i - netWidth + l;
              petscWrap->m_myPDE->JacLHSOpAsplit(
                inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmpTmp, compRow, compCol, 1);
              vTmp[l] += my1dStenGrid[l] + vTmpTmp[l];
            }
          }

          // Compute second derivative contributions if available
          if (my2dStenGrid.has_value()) {
            petscWrap->m_myPDE->JacLHSOpAsplit(
              inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmp, compRow, compCol, 0);

            // Add time derivative shift to diagonal
            if (compRow == compCol) {
              vTmp[netWidth] += shift;
            }

            // Compute off-diagonal Jacobian contributions from spatial stencil
            for (PetscInt l = 0; l < stencilWidth; ++l) {
              colTmp[l].c = compCol;
              colTmp[l].i = i - netWidth + l;
              petscWrap->m_myPDE->JacLHSOpAsplit(
                inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmpTmp, compRow, compCol, 1);
              vTmp[l] += my1dStenGrid[l] + vTmpTmp[l];
            }
          }

          PetscCall(
            MatSetValuesStencil(P, 1, &rowTmp, 3, colTmp.data(), vTmp.data(), INSERT_VALUES));
        }
      }
    }
  }
  else if (numDims == 2) {
    // Extract 2D stencil parameters
    PetscInt const stencilWidth = stenData[3];
    PetscInt const netWidth = std : floor(stencilWidth / 2);

    // Fill in ghost cells in x direction
    PetscInt const nx = info->xm - info->xs;

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        for (PetscInt x = info->xs; x < info->xs + info->xm; ++x) {
          PetscInt const left = info->xs - 1 - l;
          aY[((left * nx) + x) * numFields + field] =
            aY[(((left + 1) * nx) + x) * numFields + field];

          PetscInt const right = info->xs + info->xm + l;
          aY[(((right - 1) * nx) + x) * numFields + field] =
            aY[(((right - 2) * nx) + x) * numFields + field];
        }
      }
    }

    // Fill ghost cells in y direction
    PetscInt const ny = info->ym - info->ys;

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        for (PetscInt y = info->ys; y < info->ys + info->ym; ++y) {
          PetscInt const left = info->ys - 1 - l;
          aY[((y * ny) + left) * numFields + field] = aY[((y * ny) + left + 1) * numFields + field];

          PetscInt const right = info->ys + info->ym + l;
          aY[((y * ny) + right - 1) * numFields + field] =
            aY[((y * ny) + right - 2) * numFields + field];
        }
      }
    }

    // Assemble Jacobian for 2D domain
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        // Extract solution vector for current grid point
        inTmp = std::vector<PetscReal>(
          aY[(i * nx + j) * numFields], aY[(i * nx + j) * numFields + numFields]);

        // Compute stencil contributions from spatial derivatives
        if (my1dStenGrid.has_value()) {
          in1dTmp = std::vector<PetscReal>(numFields, 0.0f);

          // First derivative in x-direction
          for (PetscInt l = 0; l < netWidth; ++l) {
            for (PetscInt field = 0; field < numFields; ++field) {
              in1dTmp[field] +=
                my1dStenGrid[l] * aY[((i - netWidth + l) * nx + j) * numFields + field];
            }
          }

          // First derivative in y-direction
          for (PetscInt l = 0; l < netWidth; ++l) {
            for (PetscInt field = 0; field < numFields; ++field) {
              in1dTmp[field] +=
                my1dStenGrid[l] * aY[((i * nx) + (j - netWidth + l)) * numFields + field];
            }
          }
        }

        // Set row index for current 2D point
        rowTmp.i = i;

        // Assemble Jacobian entries for all field combinations
        for (PetscInt compRow = 0; compRow < numFields; ++compRow) {
          rowTmp.c = compRow;

          for (PetscInt compCol = 0; compCol < numFields; ++compCol) {
            // Compute diagonal term (time derivative + LHS operator)
            if (my1dStenGrid.has_value()) {
              petscWrap->m_myPDE->JacLHSOpAsplit(
                inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmp, compRow, compCol, 0);

              if (compRow == compCol) {
                vTmp[netWidth] += shift;
              }

              // Compute stencil contributions from first derivative
              for (PetscInt l = 0; l < netWidth; ++l) {
                colTmp[l].c = cc;
                colTmp[l].i = i * nx + j - netWidth + l;
                petscWrap->m_myPDE->JacLHSOpAsplit(
                  inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmpTmp, compRow, compCol, 1);
                vTmp[l] += my1dStenGrid[l] * vTmpTmp[l];
              }
            }

            // Compute second derivative contributions (if available)
            if (my2dStenGrid.has_value()) {
              petscWrap->m_myPDE->JacLHSOpAsplit(
                inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmp, compRow, compCol, 0);

              // Add time derivative shift to diagonal
              if (compRow == compCol) {
                vTmp[netWidth] += shift;
              }

              // Compute off-diagonal Jacobian contributions from spatial stencil
              for (PetscInt l = 0; l < stencilWidth; ++l) {
                colTmp[l].c = compCol;
                colTmp[l].i = i - netWidth + l;
                petscWrap->m_myPDE->JacLHSOpAsplit(
                  inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmpTmp, compRow, compCol, 1);
                vTmp[l] += my1dStenGrid[l] + vTmpTmp[l];
              }
            }

            // Insert row of Jacobian into matrix
            PetscCall(MatSetValuesStencil(
              P, 1, rowTmp.data(), 3, colTmp.data(), vTmp.data(), INSERT_VALUES));
          }
        }
      }
    }
  }
  else if (numDims == 3) {
    // Extract 3D stencil parameters
    PetscInt const stencilWidth = stenData[4];
    PetscInt const netWidth = std::floor(stencilWidth / 2);

    // Fill in ghost cells in x direction
    PetscInt const nx = info->xm - info->xs;

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        for (PetscInt x = info->xs; x < info->xs + info->xm; ++x) {
          PetscInt const left = info->xs - 1 - l;
          aY[((left * nx) + x) * numFields + field] =
            aY[(((left + 1) * nx) + x) * numFields + field];

          PetscInt const right = info->xs + info->xm + l;
          aY[(((right - 1) * nx) + x) * numFields + field] =
            aY[(((right - 2) * nx) + x) * numFields + field];
        }
      }
    }

    // Fill ghost cells in y direction
    PetscInt const ny = info->ym - info->ys;

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        for (PetscInt y = info->ys; y < info->ys + info->ym; ++y) {
          PetscInt const left = info->ys - 1 - l;
          aY[((y * ny) + left) * numFields + field] = aY[((y * ny) + left + 1) * numFields + field];

          PetscInt const right = info->ys + info->ym + l;
          aY[((y * ny) + right - 1) * numFields + field] =
            aY[((y * ny) + right - 2) * numFields + field];
        }
      }
    }

    // Fill in ghost cells in z direction
    PetscInt const nz = info->zm - info->zs;

    for (PetscInt l = 0; l < netWidth; ++l) {
      for (PetscInt field = 0; field < numFields; ++field) {
        for (PetscInt z = info->zs; z < info->zs + info->zm; ++z) {
          PetscInt const left = info->zs - 1 - l;
          aY[((z * nz) + left) * numFields + field] = aY[((z * nz) + left + 1) * numFields + field];

          PetscInt const right = info->zs + info->zm + l;
          aY[((z * nz) + right - 1) * numFields + field] =
            aY[((z * nz) + right - 2) * numFields + field];
        }
      }
    }

    // Assemble Jacobian for 3D domain
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
          // Extract solution vector for current grid point
          inTmp = std::vector<PetscReal>(
            aY[(i + nx * (j + ny * k)) * numFields], aY[(i + 1 + nx * (j + ny * k)) * numFields]);

          // Compute stencil contributions from spatial derivatives
          if (my1dStenGrid.has_value()) {
            in1dTmp = std::vector<PetscReal>(numFields, 0.0f);

            // First derivative in x-direction
            for (PetscInt l = 0; l < netWidth; ++l) {
              for (PetscInt field = 0; field < numFields; ++field) {
                in1dTmp[field] +=
                  my1dStenGrid[l] * aY[((i - netWidth + l) * nx + j) * numFields + field];
              }
            }

            // First derivative in y-direction
            for (PetscInt l = 0; l < netWidth; ++l) {
              for (PetscInt field = 0; field < numFields; ++field) {
                in1dTmp[field] +=
                  my1dStenGrid[l] * aY[((i * nx) + (j - netWidth + l)) * numFields + field];
              }
            }

            // First derivative in z-direction
            for (PetscInt l = 0; l < netWidth; ++l) {
              for (PetscInt field = 0; field < numFields; ++field) {
                in1dTmp[field] += my1dStenGrid[l] * aY[((i + nx))];
              }
            }
          }

          // Set row index for current 3D point
          rowTmp.i = i;

          // Assemble Jacobian entries for all field contributions
          for (PetscInt compRow = 0; compRow < numFields; ++compRow) {
            rowTmp.c = compRow;

            for (PetscInt compCol = 0; compCol < numFields; ++compCol) {
              // Compute diagonal term (time derivative + LHS operator)
              if (my1dStenGrid.has_value()) {
                petscWrap->m_myPDE->JacLHSOpAsplit(
                  inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmp compRow, compCol, 0);

                if (compRow == compCol) {
                  vTmp[netWidth] += shift;
                }

                // Compute stencil contributions from first derivative
                for (PetscInt l = 0; l < netWidth; ++l) {
                  colTmp[l].c = cc;
                  colTmp[l].i = i * nx + j - netWidth + l;
                  petscWrap->m_myPDE->JacLHSOpAsplit(
                    inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmpTmp, compRow, compCol, 1);
                  vTmp[l] += my1dStenGrid[l] * vTmpTmp[l];
                }
              }

              // Compute second derivative contributions (if available)
              if (my2dStenGrid.has_value()) {
                petscWrap->m_myPDE->JacLHSOpAsplit(
                  inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmp, compRow, compCol, 0);

                // Add time derivative shift to diagonal
                if (compRow == compCol) {
                  vTmp[netWidth] += shift;
                }

                // Compute off-diagonal Jacobian contributions from spatial stencil
                for (PetscInt l = 0; l < stencilWidth; ++l) {
                  colTmp[l].c = compCol;
                  colTmp[l].i = i - netWidth + l;
                  petscWrap->m_myPDE->JacLHSOpAsplit(
                    inTmp, aYdot, in1dTmp, in2dTmp, in3dTmp, vTmpTmp, compRow, compCol, 1);
                  vTmp[l] += my1dStenGrid[l] + vTmpTmp[l];
                }
              }

              PetscCall(MatSetValuesStencil(
                P, 1, rowTmp.data(), 3, colTmp.data(), vTmp.data(), INSERT_VALUES));
            }
          }
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  return PETSC_SUCCESS;
}

}  // namespace fcfd::pdenumerics
