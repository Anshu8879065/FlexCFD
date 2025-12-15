#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <petsc/petsc.h>

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

struct PetscProblem
{
  int bdyCon = 0;
  int setCount = 0;
  std::vector<int> stencilData;
  std::function<PetscErrorCode(DM, Vec)> initialState;
  std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**, Vec)> formRHSFunctionLocal;
  std::function<PetscErrorCode(PetscReal, PetscReal**, Mat, Mat)> formRHSJacobianLocal;
  std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**, PetscReal**)>
    formIFunctionLocal;
  std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**, PetscReal, Mat, Mat)>
    formIJacobianLocal;
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
  /// \param[in] numOptions The number of options in the options database
  ///
  explicit PetscWrap(int numOptions) noexcept
    : PetscWrap()
  {
    m_methodProps.numOptions = numOptions;
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Constructor. Creates a PetscWrap instance with `optionNames` as the options in the
  /// options database
  /// \param[in] optionNames The set of options to write to the options database
  ///
  explicit PetscWrap(std::vector<std::string> optionNames)
    : PetscWrap()
  {
    this->SetFields(optionNames);
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
  /// \param[in] numOptions The number of options in the options database
  /// \throws PetscInitException on failure
  ///
  PetscWrap(int argc, char** argv, const char* msg, int numOptions)
    : PetscWrap(argc, argv, msg)
  {
    m_methodProps.numOptions = numOptions;
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
  /// \param[in] numOptions The number of options in the options database
  /// \throws PetscInitException on failure
  ///
  PetscWrap(int argc, char** argv, const char* msg, TSType tsType, int numOptions)
    : PetscWrap(argc, argv, msg)
  {
    m_methodProps.numOptions = numOptions;
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
  /// \param[in] optionNames The set of options to write to the options database
  /// \throws PetscInitException on failure
  ///
  PetscWrap(
    int argc, char** argv, const char* msg, TSType tsType, std::vector<std::string> optionNames)
    : PetscWrap(argc, argv, msg)
  {
    m_methodProps.tsType = tsType;
    SetFields(optionNames);
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Constructor
  /// \param[in] argc The number of command-line arguments passed to the program
  /// \param[in] argv A string containing a list of the command-line arguments passed to the program
  /// \param[in] msg A help message string describing the possible options to set to modify the
  /// program's behaviour
  /// \param[in] optionNames The set of options to write to the options database
  /// \throws PetscInitException on failure
  ///
  PetscWrap(int argc, char** argv, const char* msg, std::vector<std::string> optionNames)
    : PetscWrap(argc, argv, msg)
  {
    SetFields(optionNames);
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
  /// \brief
  /// \param[in] optionNames
  /// \returns PETSC_SUCCESS on success
  ///
  auto SetFields(std::vector<std::string> optionNames) -> PetscErrorCode
  {
    m_methodProps.numOptions = std::ssize(optionNames);  // Possible narrowing conversion

    for (PetscInt i = 0; i < m_methodProps.numOptions; ++i) {
      PetscCall(DMDASetFieldName(m_da, i, optionNames[i].c_str()));
    }

    SetInitFlag(1, FlagState::IsSet);
    m_methodProps.options = std::move(optionNames);

    return PETSC_SUCCESS;
  }

  auto InitSolMethod() -> PetscErrorCode override;
  auto InitProblem() -> PetscErrorCode;

  auto InitProblem(const PetscProblem& probinit) -> PetscErrorCode;

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
  std::vector<int> m_myStencilData;
  std::optional<std::vector<PetscReal>> m_myStencilGridd;
  std::optional<std::vector<PetscReal>> m_myStencilGriddd;
  std::optional<std::vector<PetscReal>> m_myStencilGridd3;
  std::vector<PetscReal> m_inTmp;
  std::vector<PetscReal> m_outTmp;
  std::vector<PetscReal> m_vTmp;
  std::vector<PetscReal> m_vTmpTmp;

  std::vector<MatStencil> m_colTmp;

  //!
  MatStencil m_rowTmp {};

  std::vector<PetscReal> m_indTmp;
  std::vector<PetscReal> m_outdTmp;
  std::vector<PetscReal> m_vdTmp;
  std::vector<MatStencil> m_coldTmp;
  MatStencil m_rowdTmp {};
  std::optional<std::vector<PetscReal>> m_ind2Tmp;
  std::optional<std::vector<PetscReal>> m_outd2Tmp;
  std::optional<std::vector<PetscReal>> m_vd2Tmp;
  std::optional<std::vector<MatStencil>> m_cold2Tmp;
  std::optional<MatStencil> m_rowd2Tmp;
  std::optional<std::vector<PetscReal>> m_ind3Tmp;
  std::optional<std::vector<PetscReal>> m_outd3Tmp;
  std::optional<std::vector<PetscReal>> m_vd3Tmp;
  std::optional<std::vector<MatStencil>> m_cold3Tmp;
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

template<typename PDEOptions>
auto PetscWrap<PDEOptions>::FormRHSFunctionLocal(
  DMDALocalInfo* info, PetscReal t, PetscReal* aY, PetscReal* aG, void* pwp) -> PetscErrorCode
{
  int nf = m_methodProps.numOptions;

  itertools::SquareRange<PetscInt> xrange {info->xs, info->xs + info->xm};
  itertools::SquareRange<PetscInt> yrange {
    nd >= 2 ? info->ys : 0, nd >= 2 ? info->ys + info->ym : 0};
  itertools::SquareRange<PetscInt> zrange {
    nd >= 3 ? info->zs : 0, nd >= 3 ? info->zs + info->zm : 0};
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

  auto const& numOptions = std::ssize(petscWrap->m_methodProps.options);
  auto const& rowTmp = petscWrap->m_rowTmp;
  auto const& colTmp = petscWrap->m_colTmp;
  auto const& inTmp = petscWrap->m_inTmp;
  auto const& vTmp = petscWrap->m_vTmp;

  if (numDims == 1) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      rowTmp.i = i;

      for (PetscInt field = 0; field < numOptions; ++field) {
        colTmp[field].i = i;
        colTmp[field].c = field;

        rowTmp.c = field;
        inTmp = std::vector(aY[i * numOptions], aY[(i + 1) * numOptions]);
        petscWrap->m_myPDE->JacRHS(inTmp, vTmp, field);
        PetscCall(MatSetValuesStencil(P, 1, &rowTmp, 2, colTmp.data(), vTmp.data(), INSERT_VALUES));
      }
    }
  }
  else if (numDims == 2) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      rowTmp.i = i;
      PetscInt ni = info->ym - info->ys;

      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        rowTmp.j = j;

        for (PetscInt field = 0; field < numOptions; ++field) {
          colTmp[field].i = i;
          colTmp[field].j = j;
          colTmp[field].c = field;

          rowTmp.c = field;
          inTmp = std::vector(aY[(i + ni * j) * numOptions], aY[(i + 1 + ni * j) * numOptions]);
          petscWrap->m_myPDE->JacRHS(inTmp, vTmp, field);
          PetscCall(
            MatSetValuesStencil(P, 1, &rowTmp, 2, colTmp.data(), vTmp.data(), INSERT_VALUES));
        }
      }
    }
  }
  else if (numDims == 3) {
    for (PetscInt i = info->xs; i < info->xs + info->xm; ++i) {
      PetscInt ni = info->ym - info->ys;
      rowTmp.i = i;

      for (PetscInt j = info->ys; j < info->ys + info->ym; ++j) {
        rowTmp.j = j;
        PetscInt nj = info->zm - info->zs;

        for (PetscInt k = info->zs; k < info->zs + info->zm; ++k) {
          rowTmp.k = k;

          for (PetscInt field = 0; field < numOptions; ++field) {
            colTmp[field].i = i;
            colTmp[field].j = j;
            colTmp[field].k = k;
            colTmp[field].c = field;

            rowTmp.c = field;
            inTmp = std::vector(aY[(i + ni * (j + nj * k)) * numOptions],
              aY[(i + 1 + ni * (j + nj * k)) * numOptions]);
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

template<typename PDEOptions>
auto PetscWrap<PDEOptions>::FormIFunctionLocal(
  DMDALocalInfo* info, PetscReal t, PetscReal* aY, PetscReal* aYdot, PetscReal* aF, void* pwp)
  -> PetscErrorCode
{
}

template<typename PDEOptions>
auto PetscWrap<PDEOptions>::FormIJacobianLocal(DMDALocalInfo* info,
  PetscReal t,
  PetscReal* aY,
  PetscReal* aYdot,
  PetscReal shift,
  Mat J,
  Mat P,
  void* pwp) -> PetscErrorCode
{
}

}  // namespace fcfd::pdenumerics
