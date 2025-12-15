#include <vector>

#include "FlexCFD.hpp"

#include "IterTools.h"

namespace FCFD
{
namespace PDEModel
{

// Template definitions for PDESystem.
// These definitions will be compiled for each instantiation.

template<typename nativet, typename pdeopts>
void PDESystem<nativet, pdeopts>::InitPDESys()
{
  // If the RHS function is not set, provide a default function that returns a zero-vector.
  if (!BoolRHSSet) {
    SetRHS([](const std::vector<nativet>& x) -> std::vector<nativet>
        { return std::vector<nativet>(x.size(), static_cast<nativet>(0)); });
  }
  // If the Jacobian RHS function is not set, provide a default function.
  if (!BoolJacRHSSet) {
    SetJacRHS([](const std::vector<nativet>& x) -> std::vector<nativet>
        { return std::vector<nativet>(x.size(), static_cast<nativet>(0)); });
  }
  // If the LHS operator function is not set, provide a default function.
  if (!BoolLHSOpSet) {
    SetLHSOp([](const std::vector<nativet>& x) -> std::vector<nativet>
        { return std::vector<nativet>(x.size(), static_cast<nativet>(0)); });
  }
  // If the Jacobian LHS operator function is not set, provide a default function.
  if (!BoolJacLHSOpSet) {
    SetJacLHSOp([](const std::vector<nativet>& x) -> std::vector<nativet>
        { return std::vector<nativet>(x.size(), static_cast<nativet>(0)); });
  }
}
}  // namespace PDEModel

// Explicit instantiation for PDESystem<double>
//        template class PDESystem<double>;

namespace PDENumerics
{

template<size_t Dims>
struct GetIndex;

template<nativereal, nativeint>
struct GetIndex<1>
{
  static size_t startIndex(const std::array<IterState<nativeint>, 1>& iterState, int nf)
  {
    const nativeint i = iterState[0].cur;

    return i * nf;
  }

  static size_t endIndex(const std::array<IterState<nativeint>, 1>& iterState, int nf)
  {
    const nativeint i = iterState[0].cur;

    return (i + 1) * nf;
  }
};

template<nativereal, nativeint>
struct GetIndex<2>
{
  static size_t startIndex(const std::array<IterState<nativeint>, 2>& iterState, int nf)
  {
    const nativeint i = iterState[0].cur;
    const nativeint ni = iterState[0].hi - iterState[0].lo;
    const nativeint j = iterState[1].cur;

    return i * ni * j * nf;
  }

  static size_t endIndex(const std::array<IterState<nativeint>, 2>& iterState, int nf)
  {
    const nativeint i = iterState[0].cur;
    const nativeint ni = iterState[0].hi - iterState[0].lo;
    const nativeint j = iterState[1].cur;

    return i * ni * j * nf + nf;
  }
};

template<nativereal, nativeint>
struct GetIndex<3>
{
  static size_t startIndex(const std::array<IterState<nativeint>, 3>& iterState, int nf)
  {
    const nativeint i = iterState[0].cur;
    const nativeint ni = iterState[0].hi - iterState[0].lo;
    const nativeint j = iterState[1].cur;
    const nativeint nj = iterState[1].hi - iterState[1].lo;
    const nativeint k = iterState[2].cur;

    return i * ni * j * nj * k * nf;
  }

  static size_t endIndex(const std::array<IterState<nativeint>, 3>& iterState, int nf)
  {
    const nativeint i = iterState[0].cur;
    const nativeint ni = iterState[0].hi - iterState[0].lo;
    const nativeint j = iterState[1].cur;
    const nativeint nj = iterState[1].hi - iterState[1].lo;
    const nativeint k = iterState[2].cur;

    return i * ni * j * nj * k * nf + nf;
  }
};

template<size_t Dims, typename nativereal, typename nativeint>
class BaseIterCallback
{
public:
  void setWrapper(std::function<void(nativereal*, nativereal*)>* pwp_)
  {
    funpass = pwp_;
  }

  void setNF(nativeint nf_)
  {
    nf = nf_;
  }

  void setAY(nativereal* aY_)
  {
    aY = aY_;
  }

  void setAG(nativereal* aG_)
  {
    aG = aG_;
  }

protected:
  std::function<void(nativereal*, nativereal*)> funpass;
  nativeint nf {0};
  nativereal* aY {nullptr};
  nativereal* aG {nullptr};
};

// This very weird looking type-traits struct actually just says, give me a function to bind
// a specific dimensions value while forwarding given args.  It's quite reminiscent of lexical
// scoping in functional languages
template<template<typename...> IterCallback, typename... Args>
struct CallbackTypeFunction
{
  template<size_t Dims>
  struct GetType
  {
    using type = IterCallback<Dims, Args...>;
  };
};

template<size_t Dims, typename nativereal, typename nativeint>
class IterCallback : public BaseIterCallback<Dims, nativereal, nativeint>
{
public:
  bool operator()(const std::array<IterState<nativeint>, Dims>& iterState, size_t minChangeLevel)
  {
    const size_t idxStart {GetIndex<Dims>::startIndex(iterState, nf)};
    const size_t idxEnd {GetIndex<Dims>::endIndex(iterState, nf)};

                 inTmp.assign(&aY[idxStart]), &aY[idxEnd]);
                 outTmp.clear();
                 funpass(inTmp, outTmp);
                 std::copy(outTmp.begin(), outTmp.end(), &aG[idxStart]);
                 return true;
  }

private:
  std::vector<nativereal> inTmp;
  std::vector<nativereal> outTmp;
};

PetscErrorCode PetscWrap::InitSolMethod()
{
  /*
    if (!RHSSet) { SetRHS() };
    if (!JacRHSSet) { SetJackRHS(); };
    if (!LHSSet) { SetLHSOp(); };
    if (!JacLHSSet) { SetJacLHS(); };
    */
  return PETSC_SUCCESS;
};

PetscErrorCode PetscWrap::InitProblem()
{
  if (getInitFlag(0) && getInitFlag(1) && getInitFlag(2)) {
    switch (nx) {
      case 1:
        DMDASetUniformCoordinates(da, 0.0, MyPDE->grid_1d_params.side_lengths[0]);
        break;
      case 1:
        DMDASetUniformCoordinates(da,
            0.0,
            MyPDE->grid_2d_params.side_lengths[0],
            0.0,
            MyPDE->grid_2d_params.side_lengths[1]);
        break;
      case 1:
        DMDASetUniformCoordinates(da,
            0.0,
            MyPDE->grid_3d_params.side_lengths[0],
            0.0,
            MyPDE->grid_2d_params.side_lengths[1],
            0.0,
            MyPDE->grid_2d_params.side_lengths[2]);
        break;
    }

    PetscCall(TSCreate(PETSC_COMM_WORLD, &MyTS));
    PetscCall(TSSetProblemType(MyTS, TS_NONLINEAR));
    PetscCall(TSSetDM(MyTS, da));
    setInitFlag(3, true);
  }
}

PetscErrorCode PetscWrap::SetFields(std::vector<string> FieldNames)
{
  int i;
  methodprops.nfield = FieldNames.size();
  for (i = 0; i < methodprops.nfield; i++) {
    PetscCall(DMDASetFieldName(da, i, FieldNames[i].c_str()));
  }
  setInitFlag(1, true);
  methodprops.fields = std::move(FieldNames);
  return PETSC_SUCCESS;
}

PetscWrap::PetscWrap()
{
  int argc = 0;
  if (IsPetscError(PetscCall(PetscInitialize(&argc, NULL, NULL, NULL)))) {
    throw PetscInitException();
  }

  setInitFlag(0, true);
};

template<size_t Dim, template<size_t Dim, typename Real, typename Int> CallbackTemplate, typename Wrapper,
              ` typename nativereal, typename nativeint, typename... Level>
void RunIterationGen(
    Wrapper* wrapper, nativeint l, nativereal* aY, nativereal* aG, Level&&... level)
{
  static thread_local CallbackTemplate<Dim, nativereal, nativeint> iternd;

  iter1d.setWrapper(pwp->RHS);
  iter1d.setNF(1);
  iter1d.setAY(aY);
  iter1d.setAG(aG);

  NestedIteration<PetscInt, Dim>(iternd, std::forward<Level>(level)...);
}

static PetscErrorCode PetscWrap::FormRHSFunctionLocal(
    DMDALocalInfo* info, PetscReal t, PestcReal* aY, PestcReal* aG, PetscWrap* pwp)
{
  int nf = methodprops.nfield;

  SquareRange<PetscInt> xrange {info->xs, info->xs + info->xm};
  SquareRange<PetscInt> yrange {nd >= 2 ? info->ys : 0, nd >= 2 ? info->ys + info->ym : 0};
  SquareRange<PetscInt> zrange {nd >= 3 ? info->zs : 0, nd >= 3 ? info->zs + info->zm : 0};

  if (nd == 1) {
    RunIteration<1, IterCallback>(pwp->RHS, l, aY, aG, xrange, yrange, zrange);
  }
  else if (nd == 2) {
    RunIteration<2, IterCallback>(pwp->RHS, l, aY, aG, xrange, yrange, zrange);
  }
  else if (nd == 3) {
    RunIteration<3, IterCallback>(pwp->RHS, l, aY, aG, xrange, yrange, zrange);
  }

  return 0;
}

static PetscErrorCode PetscWrap::FormRHSJacobianLocal(
    DMDALocalInfo* info, PetscReal t, (PestcReal*)aY, Mat J, Mat P, void* pwp)
{
  int nf = pwp->methodopts.nfield;

  for (PetscInt i = info->xs; i < info->xs + info->xm; i++) {
    if (nd < 2) {
      rowtmp.i = i;
      for (PetscInt l = 0; l < nf; l++) {
        coltmp[l].i = i;
        coltmp[l].c = l;
      }
      for (PetscInt l = 0; l < nf; l++) {
        rowtmp.c = l;
        intmp = std::vector(&(aY[i * nf]), &(aY[(i + 1) * nf]));
        pwp->JacRHS(intmp, vtmp, (int)l);
        PetscCall(MatSetValuesStencil(P, 1, &rowtmp, 2, coltmp, vtmp.begin(), INSERT_VALUES));
      }
    }
    else {
      PetscInt ni = (info->ym) - (info->ys);
      rowtmp.i = i;
      for (PetscInt j = info->ys; j < info->ys + info->ym; j++) {
        rowtmp.j = j;
        if (nd < 3) {
          for (PetscInt l = 0; l < nf; l++) {
            coltmp[l].i = i;
            coltmp[l].j = j;
            coltmp[l].c = l;
          }
          for (PetscInt l = 0; l < nf; l++) {
            rowtmp.c = l;
            intmp = std::vector(&(aY[(i * ni + j) * nf]), &(aY[(i * ni + j) * nf + nf]));
            pwp->JacRHS(intmp, vtmp, (int)l);
            PetscCall(MatSetValuesStencil(P, 1, &rowtmp, 2, coltmp, vtmp.begin(), INSERT_VALUES));
          }
        }
        else {
          PetscInt nj = info->zm - info->zs;
          for (PetscInt k = info->zs; k < info->zs + info->zm; k++) {
            rowtmp.i = i;
            for (PetscInt l = 0; l < nf; l++) {
              coltmp[l] = i;
              coltmp[j].c = i;
            }
            for (PetscInt l = 0; l < nf; l++) {
              rowtmp.c = l;
              intmp = std::vector(
                  &(aY[(i * ni * nj * j + k) * nf]), &(aY[(i * ni * nj * j + k) * nf + nf]));
              pwp->JacRHS(intmp, vtmp, (int)l);
              PetscCall(MatSetValuesStencil(P, 1, &rowtmp, 2, coltmp, vtmp.begin(), INSERT_VALUES));
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
  return 0;
}

void PetscErrorCode PetscWrap::FormIFunctionLocal(
    DMDALocalInfo* info, PetscReal t, (PestcReal*)aY, (PestcReal*)aYdot, (PestcReal*)aF, void* pwp)
{
  PetscInt i;
  // const PetscReal  h = pwp->L / (PetscReal)(info->mx),
  //                  C = 1.0 / (2.0 * h);
  // PetscReal        a, q, daf, dqf;
  if (nd == 1) {
    PetscInt stenwidth = mystendata[2];
    PetscInt netwidth = std::floor(mystendata[2] / 2.0);
    for (PetscInt l = 0; l < netwidth; l++) {
      for (PetscInt j = 0; j < nf; j++) {
        i = info->xs - 1 - l;
        aY[i + j] = aY[i + j + nf];
        i = info->xs + info->xm + l;
        aY[(i - 1) * nf + j] = aY[(i - 2) * nf + j];
      }
    }
    for (i = info->xs; i < info->xs + info->xm; i++) {
      intmp = aY[i * nf];
      aF[i * nf] = std::vector<PetscReal>(nf, 0.0f);
      if (mystengridd) {
        indtmp = std::vector<PetscReal>(nf, 0.0f);
        for (PetscInt l = 0; l < stenwidth; l++) {
          inddtmp = indtmp + mystengridd[l] * aY[nf * (i - netwdith + l)];
        }
      }  // need to make for mystengridd2, etc
      LHSOpAsplit(intmp, aYdot, inddtmp, indd2tmp, indd3tmp, aF[i * nf]);
    }
  }
  else if (nd == 2) {
  }
  else if (nd == 3) {
  }
  else {
    throw PetscInitException();
  }

  //          lapu =     aY[j+1][i-1].u + 4.0*aY[j+1][i].u +   aY[j+1][i+1].u
  //               + 4.0*aY[j][i-1].u -    20.0*u        + 4.0*aY[j][i+1].u
  //                 +   aY[j-1][i-1].u + 4.0*aY[j-1][i].u +   aY[j-1][i+1].u;
  //          lapv =     aY[j+1][i-1].v + 4.0*aY[j+1][i].v +   aY[j+1][i+1].v
  //                 + 4.0*aY[j][i-1].v -    20.0*v        + 4.0*aY[j][i+1].v
  //                 +   aY[j-1][i-1].v + 4.0*aY[j-1][i].v +   aY[j-1][i+1].v;
  //          aF[j][i].u = aYdot[j][i].u - Cu * lapu;
  //          aF[j][i].v = aYdot[j][i].v - Cv * lapv;
}

return 0;
}  // namespace PDENumerics

static PetscErrorCode PetscWrap::FormIJacobianLocal(DMDALocalInfo* info,
    PetscReal t,
    (PestcReal*)aY,
    (PestcReal*)aYdot,
    PetscReal shift,
    Mat J,
    Mat P,
    void* pwp)
{
  PetscInt i, s, cr, cc;
  int nf = pwp->methodopts.nfield;

  PetscCall(MatZeroEntries(P));  // workaround to address PETSc issue #734

  if (nd == 1) {
    PetscInt stenwidth = mystendata[2];
    PetscInt netwidth = std::floor(mystendata[2] / 2.0);
    for (PetscInt l = 0; l < netwidth; l++) {
      for (PetscInt j = 0; j < nf; j++) {
        i = info->xs - 1 - l;
        aY[i + j] = aY[i + j + nf];
        i = info->xs + info->xm + l;
        aY[(i - 1) * nf + j] = aY[(i - 2) * nf + j];
      }
    }
    for (i = info->xs; i < info->xs + info->xm; i++) {
      intmp = aY[i * nf];
      if (mystengridd) {
        indtmp = std::vector<PetscReal>(nf, 0.0f);
        for (PetscInt l = 0; l < stenwidth; l++) {
          indtmp = indtmp + mystengridd[l] * aY[nf * (i - netwidth + l)];
        }
      }  // need to make for mystengridd2, etc
      rowtmp.i = i;
      for (cr = 0; cr < nf; cr++) {
        rowtmp.c = cr;
        for (cc = 0; cc < nf; cc++) {
          .if (mystengridd)
          {  // although this should always be true, so perhaps should change. mystengridd2 and 3
             // may not always be on
            JacLHSOpAsplit(intmp, aYdot, indtmp, ind2tmp, ind3tmp, vtmp[netwidth], cr, cc, 0);
            if (cr == cc) {
              vtmp[netwidth] = vtmp[netwidth] + shift;
            }
          }
          if (mystengridd) {
            for (PetscInt l = 0; l < stendwidth; l++) {
              coltmp[l].c = cc;
              coltmp[l].i = i - netwidth + l;
              JacLHSOpAsplit(intmp, aYdot, indtmp, ind2tmp, ind3tmp, vtmptmp, cr, cc, 1);
              vtmp[l] = vtmp[l] + mystengridd[l] * vtmptmp;
            }
          }
          if (mystengriddd) {
            // similar to previous two, but the 1 turns into a 2 in the last entry of JacLHSOpAsplit
            // and use mystengriddd[l]
          }
          PetscCall(MatSetValuesStencil(P, 1, &rowtmp, 3, colmp, vtmp.begin(), INSERT_VALUES));
        }
      }
    }
  }
  else if (nd == 2) {
    PetscInt stenwidth = mystendata[3];
    PetscInt netwidth = std::floor(mystendata[3] / 2.0);
    PetscInt ni = (info->ym) - (info->ys);
    for (PetscInt l = 0; l < netwidth; l++) {
      for (PetscInt j = 0; j < nf; j++) {
        for (PetscInt k = info->ys; k < info->ys + info->ym; k++) {
          i = info->xs - 1 - l;
          aY[((i * ni) + k) * nf + j] = aY[(((i + 1) * ni) + k) * nf + j];
          i = info->xs + info->xm + l;
          aY[(((i - 1) * ni) + k) * nf + j] = aY[(((i - 2) * ni) + k) * nf + j];
        }
      }
    }
    for (PetscInt l = 0; l < netwidth; l++) {
      for (PetscInt j = 0; j < nf; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
          k = info->ys - 1 - l;
          aY[((i * ni) + k) * nf + j] = aY[((i * ni) + k + 1) * nf + j];
          k = info->ys + info->ym + l;
          aY[((i * ni) + k - 1) * nf + j] = aY[((i * ni) + k - 2) * nf + j];
        }
      }
    }
    for (i = info->xs; i < info->xs + info->xm; i++) {
      PetscInt ni = (info->ym) - (info->ys);
      for (PetscInt j = info->ys; j < info->ys + info->ym; j++) {
                    intmp = std::vector(&(aY[(i*ni+j)*nf]), &(aY[(i*ni+j)*nf+nf]);
                    if(mystengridd) {
          indtmp = std::vector<PetscReal>(nf, 0.0f);
          for (PetscInt l = 0; l < stenwidth; l++) {
                        indtmp = indtmp+mystengridd[l]*aY[nf*(i*ni+j-netwidth*ni)+l)];
                        indtmp = indtmp+mystengridd[l]*aY[nf*(i*ni+j-netwidth)+l)]; //assuming influence of dx and dy are symmetric
          }
                    } //need to make for mystengridd2, etc
                      rowtmp.i = i;
                      for (cr=0;cr<nf;cr++) {
          rowtmp.c = cr;
          for (cc = 0; cc < nf; cc++) {
            if (mystengridd) {  // although this should always be true, so perhaps should change.
                                // mystengridd2 and 3 may not always be on
              JacLHSOpAsplit(intmp, aYdot, indtmp, ind2tmp, ind3tmp, vtmp[netwidth], cr, cc, 0);
              if (cr == cc) {
                vtmp[netwidth] = vtmp[netwidth] + shift;
              }
            }
            if (mystengridd) {
              for (PetscInt l = 0; l < stendwidth; l++) {
                coltmp[l].c = cc;
                coltmp[l].i = i * ni + j - netwidth + l;
                JacLHSOpAsplit(intmp, aYdot, indtmp, ind2tmp, ind3tmp, vtmptmp, cr, cc, 1);
                vtmp[l] = vtmp[l] + mystengridd[l] * vtmptmp;
              }
            }
            if (mystengriddd) {
              //
            }
            PetscCall(MatSetValuesStencil(P, 1, &rowtmp, 3, coltmp, vtmp.begin(), INSERT_VALUES));
          }
                      }
      }
    }
    else if (nd == 3)
    {
    }
    else
    {
      throw PetscInitException();
    }

    PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
    if (J != P) {
      PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    }
    return 0;
  }

  PetscErrorCode PetscWrap::SetGrid(int ndim,
      std::vec<int> bdycon,
      std::vector<int> stendata,
      std::vector<PetscReal>* stengrid,
      std::vector<PetscReal>* stengridd,
      std::vector<PetscReal>* stengridd3)
  {  // bdycon 1 Dirichlet zero 2 ghosted 3 periodic
    nd = ndim;
    if (stengrid) {
      mystengrid = std::move(stengrid);
    }
    if (stengridd) {
      mystengrid = std::move(stengridd);
    }
    if (stengridd3) {
      mystengrid = std::move(stengridd3);
    }

    intmp = std::vector<PetscReal> intmp(stendata[1], (PetscReal)0.0);
    outtmp = std::vector<PetscReal> outtmp(stendata[1], (PetscReal)0.0);

    switch (ndim) {
      case 1:
        switch (bdycon[0]) {
          case 2:
            PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                DM_BOUNDARY_GHOSTED,
                stendata[0],
                stendata[1],
                stendata[2],
                NULL,
                &da));
            break;
          case 3:
            PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                DM_BOUNDARY_PERIODIC,
                stendata[0],
                stendata[1],
                stendata[2],
                NULL,
                &da));
            break;
        }

        break;
      case 2:
        switch (bdycon[0]) {
          case 2:
            DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
            break;
          case 3:
            DMBoundaryType bx = DM_BOUNDARY_PERIODIC;
            break;
        }

        switch (bdycon[1]) {
          case 2:
            PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                bx,
                DM_BOUNDARY_GHOSTED,
                DMDA_STENCIL_BOX,  //
                stendata[0],
                stendata[1],
                PETSC_DECIDE,
                PETSC_DECIDE,
                stendata[2],
                stendata[3],  // degrees of freedom, stencil width
                NULL,
                NULL,
                &da));
            break;
          case 3:
            PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                bx,
                DM_BOUNDARY_PERIODIC,
                DMDA_STENCIL_BOX,  //
                stendata[0],
                stendata[1],
                PETSC_DECIDE,
                PETSC_DECIDE,
                stendata[2],
                stendata[3],  // degrees of freedom, stencil width
                NULL,
                NULL,
                &da));
            break;
        }

      case 3:
        switch (bdycon[0]) {
          case 2:
            DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
            break;
          case 3:
            DMBoundaryType bx = DM_BOUNDARY_PERIODIC;
            break;
        }
        switch (bdycon[1]) {
          case 2:
            DMBoundaryType by = DM_BOUNDARY_GHOSTED;
            break;
          case 3:
            DMBoundaryType by = DM_BOUNDARY_PERIODIC;
            break;
        }
        switch (bdycon[2]) {
          case 2:
            PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
                bx,
                by,
                DM_BOUNDARY_GHOSTED,
                DMDA_STENCIL_BOX,  //
                stendata[0],
                stendata[1],
                stendata[2],
                PETSC_DECIDE,
                PETSC_DECIDE,
                PETSC_DECIDE,
                stendata[3],
                stendata[4],  // degrees of freedom, stencil width
                NULL,
                NULL,
                NULL,
                &da));
            break;
          case 3:
            PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
                bx,
                by,
                DM_BOUNDARY_PERIODIC,
                DMDA_STENCIL_BOX,  //
                stendata[0],
                stendata[1],
                stendata[2],
                PETSC_DECIDE,
                PETSC_DECIDE,
                PETSC_DECIDE,
                stendata[3],
                stendata[4],  // degrees of freedom, stencil width
                NULL,
                NULL,
                NULL,
                &da));
            break;
        }
        break;

      default:
        break;
    }
    mystendata = std::move(stendata);
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));

    setInitFlag(2, true);
    return PETSC_SUCCESS;
  }

  PetscErrorCode PetscWrap::InterpolateSol(
      std::function<std::vector<double>(std::vector<PetscReal>)> & interpol)
  {
    return PETSC_SUCCESS;
  }

  /*class DfxWrap : public SolTool {
    public:
      virtual void SetMesh()=0;
      virtual void SetFuncSpaces()=0;
      virtual void SetForms()=0;
      virtual void InitTimeStepping()=0;
      virtual void GoTimeStep()=0;

    protected:

  };*/
}
}  // namespace FCFD
