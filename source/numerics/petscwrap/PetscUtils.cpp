#include <cassert>
#include <utility>
#include <petscsys.h>   // PetscCall, PetscFunctionBeginUser, PetscFunctionReturn, PetscPrintf
#include <petscdm.h>    // DM
#include <petscdmda.h>  // DMDA helpers, DMDAVecGetArrayDOF*, DMDACreate*
#include <petscvec.h>   // Vec

#include "PetscUtils.hpp"

namespace fcfd::pdenumerics
{

auto FillGhostCells1d(DM const& distributedMesh, DMDALocalInfo const& info) -> PetscErrorCode
{
  assert(info.bx == DM_BOUNDARY_GHOSTED);

  PetscFunctionBeginUser;

  Vec localVec {};
  PetscReal* array = nullptr;

  PetscCall(DMGetLocalVector(distributedMesh, &localVec));
  PetscCall(DMDAVecGetArray(distributedMesh, localVec, &array));

  for (PetscInt i = info.gxs; i < info.gxs + info.gxm; ++i) {
    if (i < 0) {
      array[i] = array[i + 1];
    }

    if (i >= info.xm) {
      array[i] = array[i - 1];
    }
  }

  PetscCall(DMDAVecRestoreArray(distributedMesh, localVec, &array));
  PetscCall(DMRestoreLocalVector(distributedMesh, &localVec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

auto FillGhostCells2d(DM const& distributedMesh, DMDALocalInfo const& info) -> PetscErrorCode
{
  assert(info.bx == DM_BOUNDARY_GHOSTED and info.by == DM_BOUNDARY_GHOSTED);

  PetscFunctionBeginUser;

  Vec localVec {};
  PetscReal** array = nullptr;

  PetscCall(DMGetLocalVector(distributedMesh, &localVec));
  PetscCall(DMDAVecGetArray(distributedMesh, localVec, &array));

  for (PetscInt j = info.gys; j < info.gys + info.gym; ++j) {
    for (PetscInt i = info.gxs; i < info.gxs + info.gxm; ++i) {
      if (i < 0) {
        array[j][i] = array[j][i + 1];
      }

      if (i >= info.xm) {
        array[j][i] = array[j][i - 1];
      }

      if (j < 0) {
        array[j][i] = array[j + 1][i];
      }

      if (j > info.ym) {
        array[j][i] = array[j - 1][i];
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(distributedMesh, localVec, &array));
  PetscCall(DMRestoreLocalVector(distributedMesh, &localVec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

auto FillGhostCells3d(DM const& distributedMesh, DMDALocalInfo const& info) -> PetscErrorCode
{
  assert(info.bx == DM_BOUNDARY_GHOSTED and info.by == DM_BOUNDARY_GHOSTED and info.bz == DM_BOUNDARY_GHOSTED);

  PetscFunctionBeginUser;

  Vec localVec {};
  PetscReal*** array = nullptr;

  PetscCall(DMGetLocalVector(distributedMesh, &localVec));
  PetscCall(DMDAVecGetArray(distributedMesh, localVec, &array));

  for (PetscInt k = info.gzs; k < info.gzs + info.gzm; ++k) {
    for (PetscInt j = info.gys; j < info.gys + info.gym; ++j) {
      for (PetscInt i = info.gxs; i < info.gxs + info.gxm; ++i) {
        if (i < 0) {
          array[k][j][i] = array[k][j][i + 1];
        }

        if (i >= info.xm) {
          array[k][j][i] = array[k][j][i - 1];
        }

        if (j < 0) {
          array[k][j][i] = array[k][j + 1][i];
        }

        if (j >= info.ym) {
          array[k][j][i] = array[k][j - 1][i];
        }

        if (k < 0) {
          array[k][j][i] = array[k + 1][j][i];
        }

        if (k >= info.zm) {
          array[k][j][i] = array[k - 1][j][i];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(distributedMesh, localVec, &array));
  PetscCall(DMRestoreLocalVector(distributedMesh, &localVec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

auto Create1dStencilGrid(DM& dmPtr, std::span<PetscInt> stencilData, DMBoundaryType const boundaryCond)
  -> PetscErrorCode
{
  assert(stencilData.size() >= 3 and "stencilData is missing some entries");
  assert((boundaryCond == DM_BOUNDARY_GHOSTED or boundaryCond == DM_BOUNDARY_PERIODIC) and "Unsupported boundary type");

  if (boundaryCond == DM_BOUNDARY_GHOSTED) {
    PetscCall(DMDACreate1d(
      PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, stencilData[0], stencilData[1], stencilData[2], nullptr, &dmPtr));
  }
  else if (boundaryCond == DM_BOUNDARY_PERIODIC) {
    PetscCall(DMDACreate1d(
      PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, stencilData[0], stencilData[1], stencilData[2], nullptr, &dmPtr));
  }

  return 0;
}

auto Create2dStencilGrid(DM& dmPtr, std::span<PetscInt> stencilData, DMBoundaryType const boundaryCond)
  -> PetscErrorCode
{
  assert(stencilData.size() >= 4 and "stencilData is missing some entries");
  assert((boundaryCond == DM_BOUNDARY_GHOSTED or boundaryCond == DM_BOUNDARY_PERIODIC) and "Unsupported boundary type");

  if (boundaryCond == DM_BOUNDARY_GHOSTED) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_GHOSTED,
      DM_BOUNDARY_GHOSTED,
      DMDA_STENCIL_BOX,
      stencilData[0],
      stencilData[1],
      PETSC_DECIDE,
      PETSC_DECIDE,
      stencilData[2],
      stencilData[3],
      nullptr,
      nullptr,
      &dmPtr));
  }
  else if (boundaryCond == DM_BOUNDARY_PERIODIC) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_PERIODIC,
      DM_BOUNDARY_PERIODIC,
      DMDA_STENCIL_BOX,
      stencilData[0],
      stencilData[1],
      PETSC_DECIDE,
      PETSC_DECIDE,
      stencilData[2],
      stencilData[3],
      nullptr,
      nullptr,
      &dmPtr));
  }

  return 0;
}

auto Create3dStencilGrid(DM& dmPtr, std::span<PetscInt> stencilData, DMBoundaryType const boundaryCond)
  -> PetscErrorCode
{
  assert(stencilData.size() >= 5 and "stencilData is missing some entries");
  assert((boundaryCond == DM_BOUNDARY_GHOSTED or boundaryCond == DM_BOUNDARY_PERIODIC) and "Unsupported boundary type");

  if (boundaryCond == DM_BOUNDARY_GHOSTED) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
      DM_BOUNDARY_GHOSTED,
      DM_BOUNDARY_GHOSTED,
      DM_BOUNDARY_GHOSTED,
      DMDA_STENCIL_BOX,
      stencilData[0],
      stencilData[1],
      stencilData[2],
      PETSC_DECIDE,
      PETSC_DECIDE,
      PETSC_DECIDE,
      stencilData[3],
      stencilData[4],
      nullptr,
      nullptr,
      nullptr,
      &dmPtr));
  }
  else if (boundaryCond == DM_BOUNDARY_PERIODIC) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
      DM_BOUNDARY_PERIODIC,
      DM_BOUNDARY_PERIODIC,
      DM_BOUNDARY_PERIODIC,
      DMDA_STENCIL_BOX,
      stencilData[0],
      stencilData[1],
      stencilData[2],
      PETSC_DECIDE,
      PETSC_DECIDE,
      PETSC_DECIDE,
      stencilData[3],
      stencilData[4],
      nullptr,
      nullptr,
      nullptr,
      &dmPtr));
  }

  return 0;
}

auto SetInitialState(DM& distributedMesh,
  Vec& vec,
  DMDALocalInfo& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode
{
  PetscFunctionBeginUser;

  PetscCall(VecSet(vec, 0));
  PetscCall(DMDAGetLocalInfo(distributedMesh, &info));

  if (info.dim == 1) {
    return SetInitialState1d(distributedMesh, vec, info, initFunc);
  }
  else if (info.dim == 2) {
    return SetInitialState2d(distributedMesh, vec, info, initFunc);
  }
  else if (info.dim == 3) {
    return SetInitialState3d(distributedMesh, vec, info, initFunc);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

auto SetInitialState1d(DM const& distributedMesh,
  Vec const& vec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode
{
  assert(distributedMesh and "The DM object is in an invalida state and must be initialized before use");
  assert(vec and "The Vec object is in an invalid state and must be initialized before use");

  PetscFunctionBeginUser;

  PetscReal** fieldArray = nullptr;
  PetscCall(DMDAVecGetArrayDOF(distributedMesh, vec, &fieldArray));

  for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
    std::vector<PetscReal> coords = {static_cast<PetscReal>(i)};
    auto vals = initFunc(coords);
    assert(std::cmp_equal(info.dof, vals.size()) and "InitCond returned wrong-sized vector");

    for (std::size_t field = 0; field < info.dof; ++field) {
      fieldArray[i][field] = vals[static_cast<std::size_t>(field)];
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(distributedMesh, vec, &fieldArray));

  PetscFunctionReturn(PETSC_SUCCESS);
}

auto SetInitialState2d(DM const& distributedMesh,
  Vec const& vec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode
{
  assert(distributedMesh and "The DM object is in an invalida state and must be initialized before use");
  assert(vec and "The Vec object is in an invalid state and must be initialized before use");
  PetscFunctionBeginUser;

  PetscReal*** fieldArray = nullptr;
  PetscCall(DMDAVecGetArrayDOF(distributedMesh, vec, &fieldArray));

  for (PetscInt j = info.ys; j < info.ys + info.ym; ++j) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
      std::vector<PetscReal> coords = {static_cast<PetscReal>(i), static_cast<PetscReal>(j)};
      auto vals = initFunc(coords);
      assert(static_cast<std::size_t>(info.dof) == vals.size() && "InitCond returned wrong-sized vector");

      for (PetscInt field = 0; field < info.dof; ++field) {
        fieldArray[j][i][field] = vals[static_cast<std::size_t>(field)];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(distributedMesh, vec, &fieldArray));

  PetscFunctionReturn(PETSC_SUCCESS);
}

auto SetInitialState3d(DM const& distributedMesh,
  Vec const& vec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal> const&)> const& initFunc) -> PetscErrorCode
{
  assert(distributedMesh and "The DM object is in an invalida state and must be initialized before use");
  assert(vec and "The Vec object is in an invalid state and must be initialized before use");
  PetscFunctionBeginUser;

  PetscReal**** fieldArray = nullptr;
  PetscCall(DMDAVecGetArrayDOF(distributedMesh, vec, &fieldArray));

  for (PetscInt k = info.zs; k < info.zs + info.zm; ++k) {
    for (PetscInt j = info.ys; j < info.ys + info.ym; ++j) {
      for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
        std::vector<PetscReal> coords = {
          static_cast<PetscReal>(i), static_cast<PetscReal>(j), static_cast<PetscReal>(k)};
        auto vals = initFunc(coords);
        assert(static_cast<std::size_t>(info.dof) == vals.size() && "InitCond returned wrong-sized vector");

        for (PetscInt field = 0; field < info.dof; ++field) {
          fieldArray[k][j][i][field] = vals[static_cast<std::size_t>(field)];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(distributedMesh, vec, &fieldArray));

  PetscFunctionReturn(PETSC_SUCCESS);
}

auto EvaluateSolution(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode
{
  assert(distributedMesh and "The DM object is in an invalid state");
  assert(globalVec and "The Vec object is in an invalid state");
  assert(info.dim >= 1 and info.dim <= 3 and "Unsupported number of dimensions");

  PetscFunctionBeginUser;

  if (info.dim == 1) {
    return EvaluateSolution1d(distributedMesh, globalVec, info, func);
  }
  else if (info.dim == 2) {
    return EvaluateSolution2d(distributedMesh, globalVec, info, func);
  }
  else if (info.dim == 3) {
    return EvaluateSolution3d(distributedMesh, globalVec, info, func);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ===============================
// FIXED: call func ONCE per grid point, and DO NOT print here
// ===============================

auto EvaluateSolution1d(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode
{
  assert(distributedMesh and "The DM object is in an invalid state");
  assert(globalVec and "The Vec object is in an invalid state");
  assert(info.dim == 1 and "Unsupported number of dimensions");

  PetscFunctionBeginUser;

  auto nodeVals = std::vector<PetscReal>(info.dof, 0);
  PetscReal** fieldArray = nullptr;
  PetscCall(DMDAVecGetArrayDOFRead(distributedMesh, globalVec, &fieldArray));

  for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
    for (PetscInt field = 0; field < info.dof; ++field) {
      nodeVals[static_cast<std::size_t>(field)] = fieldArray[i][field];
    }
    (void)func(nodeVals); // ✅ once per i
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(distributedMesh, globalVec, &fieldArray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

auto EvaluateSolution2d(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode
{
  assert(distributedMesh and "The DM object is in an invalid state");
  assert(globalVec and "The Vec object is in an invalid state");
  assert(info.dim == 2 and "Unsupported number of dimensions");

  PetscFunctionBeginUser;

  auto nodeVals = std::vector<PetscReal>(info.dof, 0);
  PetscReal*** fieldArray = nullptr;
  PetscCall(DMDAVecGetArrayDOFRead(distributedMesh, globalVec, &fieldArray));

  for (PetscInt j = info.ys; j < info.ys + info.ym; ++j) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
      for (PetscInt field = 0; field < info.dof; ++field) {
        nodeVals[static_cast<std::size_t>(field)] = fieldArray[j][i][field];
      }
      (void)func(nodeVals); // ✅ once per (i,j)
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(distributedMesh, globalVec, &fieldArray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

auto EvaluateSolution3d(DM const& distributedMesh,
  Vec const& globalVec,
  DMDALocalInfo const& info,
  std::function<std::vector<PetscReal>(std::vector<PetscReal>&)> const& func) -> PetscErrorCode
{
  assert(distributedMesh and "The DM object is in an invalid state");
  assert(globalVec and "The Vec object is in an invalid state");
  assert(info.dim == 3 and "Unsupported number of dimensions");

  PetscFunctionBeginUser;

  auto nodeVals = std::vector<PetscReal>(info.dof, 0);
  PetscReal**** fieldArray = nullptr;
  PetscCall(DMDAVecGetArrayDOFRead(distributedMesh, globalVec, &fieldArray));

  for (PetscInt k = info.zs; k < info.zs + info.zm; ++k) {
    for (PetscInt j = info.ys; j < info.ys + info.ym; ++j) {
      for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
        for (PetscInt field = 0; field < info.dof; ++field) {
          nodeVals[static_cast<std::size_t>(field)] = fieldArray[k][j][i][field];
        }
        (void)func(nodeVals); // ✅ once per (i,j,k)
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(distributedMesh, globalVec, &fieldArray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace fcfd::pdenumerics
