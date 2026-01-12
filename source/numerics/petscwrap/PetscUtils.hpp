#pragma once

#include <functional>
#include <span>
#include <vector>

#include <petsc.h>

namespace fcfd::pdenumerics
{

///
/// \brief Check if a given PetscErrorCode is an error code
/// \param[in] code The code in question
/// \returns True if the code is an error code, and false otherwise
///
constexpr auto IsPetscError(PetscErrorCode const code) noexcept -> bool
{
  return code != 0;
}

struct CallBacks
{
  PetscBool iFuncCalled;
  PetscBool iJacobianCalled;
  PetscBool rhsFuncCalled;
  PetscBool rhsJacobianCalled;
};

///
/// \brief Fill in ghost cells in the x, y, and z directions
/// \param[in] info DMDALocalInfo struct containing grid indices and dimensions
/// \param[in] fieldArray Array of field variables
/// \param[in] numFields The number of field variable being processed
/// \param[in] netWidth The radius of the grid cell
///
void FillGhostCells(DMDALocalInfo const& info, std::span<PetscReal> fieldArray, std::size_t numFields, int netWidth);

///
/// \brief Fill in ghost cells in a 1d grid
/// \param[in] distributedMesh The DM object
/// \param[in] info The DMDALocalInfo struct containing grid indices and dimensions
/// \returns A PetscErrorCode indicating success or failure
///
auto FillGhostCells1d(DM const& distributedMesh, DMDALocalInfo const& info) -> PetscErrorCode;

///
/// \brief Fill in ghost cells in a 2d grid
/// \param[in] distributedMesh The DM object
/// \param[in] info The DMDALocalInfo struct containing grid indices and dimensions
/// \returns A PetscErrorCode indicating success or failure
///
auto FillGhostCells2d(DM const& distributedMesh, DMDALocalInfo const& info) -> PetscErrorCode;

///
/// \brief Fill in ghost cells in a 3d grid
/// \param[in] distributedMesh The DM object
/// \param[in] info The DMDALocalInfo struct containing grid indices and dimensions
/// \returns A PetscErrorCode indicating success or failure
///
auto FillGhostCells3d(DM const& distributedMesh, DMDALocalInfo const& info) -> PetscErrorCode;

///
/// \brief Create a 1d stencil grid
/// \param dmPtr The data manager object from which the grid is to be created
/// \param stencilData Stencil data (i.e., global dimensions in x direction, degrees of freedom per node, stencil width)
/// \param boundaryCond The type of boundary condition to apply on the grid
/// \return A PetscErrorCode indicating success or failure
///
auto Create1dStencilGrid(DM& dmPtr, std::span<PetscInt> stencilData, DMBoundaryType boundaryCond) -> PetscErrorCode;

///
/// \brief Create a 2d stencil grid
/// \param dmPtr The data manager object from which the grid is to be created
/// \param stencilData Stencil data (i.e., global dimensions in x and y directions, degrees of freedom per node, stencil
/// width)
/// \param boundaryCond The type of boundary condition to apply on the grid
/// \return A PetscErrorCode indicating success or failure
///
auto Create2dStencilGrid(DM& dmPtr, std::span<PetscInt> stencilData, DMBoundaryType boundaryCond) -> PetscErrorCode;

///
/// \brief Create a 3d stencil grid
/// \param dmPtr The data manager object from which the grid is to be created
/// \param stencilData Stencil data (i.e., global dimensions in x, y, and z directions, degrees of freedom per node,
/// stencil width)
/// \param boundaryCond The type of boundary condition to apply on the grid
/// \return A PetscErrorCode indicating success or failure
///
auto Create3dStencilGrid(DM& dmPtr, std::span<PetscInt> stencilData, DMBoundaryType boundaryCond) -> PetscErrorCode;

///
/// \brief Set the initial state of the PDE system
/// \param[in] distributedMesh The DM object
/// \param[in] vec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] initFunc A function that accepts grid indices (i[,j[,k]]) as a vector of PetscReal and
/// returns a vector containing the initial DOF values at that location (vector length must equal the DMDA DOF)
/// \returns A PetscErrorCode indicating success or failure
///
auto SetInitialState(DM& distributedMesh,
  Vec& vec,
  DMDALocalInfo& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode;

///
/// \brief Set the initial state of the PDE system (1D)
/// \param[in] distributedMesh The DM object
/// \param[in] vec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] initFunc A function that accepts grid index {i} (as a vector {i}, PetscReal)
/// and returns a vector containing the initial DOF values at that location (vector length must equal the DMDA DOF)
/// \returns A PetscErrorCode indicating success or failure
///
auto SetInitialState1d(DM const& distributedMesh,
  Vec const& vec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode;

///
/// \brief Set the initial state of the PDE system (2D)
/// \param[in] distributedMesh The DM object
/// \param[in] vec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] initFunc A function that accepts grid indices {i,j} (PetscReal) and returns
/// a vector containing the initial DOF values at that location (vector length must equal the DMDA DOF)
/// \returns A PetscErrorCode indicating success or failure
///
auto SetInitialState2d(DM const& distributedMesh,
  Vec const& vec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode;

///
/// \brief Set the initial state of the PDE system (3D)
/// \param[in] distributedMesh The DM object
/// \param[in] vec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] initFunc A function that accepts grid indices {i,j,k} (PetscReal) and returns
/// a vector containing the initial DOF values at that location (vector length must equal the DMDA DOF)
/// \returns A PetscErrorCode indicating success or failure
///
auto SetInitialState3d(DM const& distributedMesh,
  Vec const& vec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode;

///
/// \brief Evaluate the solution at a certain grid point using the given function
/// \param[in] distributedMesh The DM object
/// \param[in] globalVec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] func The function which will be used to process the solution at a certain grid point
/// \returns A PetscErrorCode indicating success or failure
///
auto EvaluateSolution(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode;

///
/// \brief Evaluate the solution at a certain grid point using the given function
/// \param[in] distributedMesh The DM object
/// \param[in] globalVec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] func The function which will be used to process the solution at a certain grid point
/// \returns A PetscErrorCode indicating success or failure
///
auto EvaluateSolution1d(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode;

///
/// \brief Evaluate the solution at a certain grid point using the given function
/// \param[in] distributedMesh The DM object
/// \param[in] globalVec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] func The function which will be used to process the solution at a certain grid point
/// \returns A PetscErrorCode indicating success or failure
///
auto EvaluateSolution2d(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode;

///
/// \brief Evaluate the solution at a certain grid point using the given function
/// \param[in] distributedMesh The DM object
/// \param[in] globalVec The Vec object
/// \param[in] info The DMDALocalInfo object
/// \param[in] func The function which will be used to process the solution at a certain grid point
/// \returns A PetscErrorCode indicating success or failure
///
auto EvaluateSolution3d(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode;
}  // namespace fcfd::pdenumerics
