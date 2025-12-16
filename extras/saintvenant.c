static char help[] =
"Coupled Saint Venant System.  Option prefix -ptn_.\n"
"Incorporates form  F(t,Y,dot Y) = G(t,Y)  where F() is IFunction and G() is\n"
"RHSFunction().  Implements IJacobian() and RHSJacobian().  Defaults to\n"
"ARKIMEX (= adaptive Runge-Kutta implicit-explicit) TS type.\n\n";

#include <petsc.h>
#include <petscmath.h>

typedef struct
{
  PetscReal a, q;
} Field;

typedef struct
{
  PetscReal L,  // domain side length
      So,  // Constant slope, later make it a function
      g,  // Constant of gravity
      nm;  // Dimension based friction constant
  PetscBool IFcn_called, IJac_called, RHSFcn_called, RHSJac_called;
} PatternCtx;

extern PetscErrorCode InitialState(DM, Vec, PetscReal, PatternCtx*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal, Field*, Field*, PatternCtx*);
extern PetscErrorCode FormRHSJacobianLocal(
    DMDALocalInfo*, PetscReal, Field*, Mat, Mat, PatternCtx*);
extern PetscErrorCode FormIFunctionLocal(
    DMDALocalInfo*, PetscReal, Field*, Field*, Field*, PatternCtx*);
extern PetscErrorCode FormIJacobianLocal(
    DMDALocalInfo*, PetscReal, Field*, Field*, PetscReal, Mat, Mat, PatternCtx*);

int main(int argc, char** argv)
{
  PatternCtx user;
  TS ts;
  Vec x;
  DM da;
  DMDALocalInfo info;
  PetscReal noiselevel = -1.0;  // negative value means no initial noise
  PetscBool no_rhsjacobian = PETSC_FALSE, no_ijacobian = PETSC_FALSE, call_back_report = PETSC_TRUE;
  TSType type;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // parameter values for now arbitrarily chosen
  user.L = 2.5;
  user.So = -1.5;
  user.g = 9.8;
  user.nm = 1.;
  user.IFcn_called = PETSC_FALSE;
  user.IJac_called = PETSC_FALSE;
  user.RHSFcn_called = PETSC_FALSE;
  user.RHSJac_called = PETSC_FALSE;
  PetscOptionsBegin(PETSC_COMM_WORLD, "ptn_", "options for patterns", "");
  PetscCall(PetscOptionsBool("-call_back_report",
      "report on which user-supplied call-backs were actually called",
      "saintvenant.c",
      call_back_report,
      &(call_back_report),
      NULL));
  PetscCall(PetscOptionsReal("-g", "gravity coefficient", "saintvenant.c", user.g, &user.g, NULL));
  PetscCall(PetscOptionsReal("-So", "Surface Slope", "saintvenant.c", user.So, &user.So, NULL));
  PetscCall(PetscOptionsReal(
      "-nm", "normal coefficient in the forcing term", "saintvenant.c", user.nm, &user.nm, NULL));
  PetscCall(PetscOptionsReal("-L",
      "square domain side length; recommend L >= 0.5",
      "saintvenant.c",
      user.L,
      &user.L,
      NULL));
  PetscCall(PetscOptionsBool("-no_ijacobian",
      "do not set call-back DMDATSSetIJacobian()",
      "saintvenant.c",
      no_ijacobian,
      &(no_ijacobian),
      NULL));
  PetscCall(PetscOptionsBool("-no_rhsjacobian",
      "do not set call-back DMDATSSetRHSJacobian()",
      "saintvenant.c",
      no_rhsjacobian,
      &(no_rhsjacobian),
      NULL));
  PetscCall(PetscOptionsReal("-noisy_init",
      "initialize a,q with this much random noise (e.g. 0.2) on top of usual initial values",
      "saintvenant.c",
      noiselevel,
      &noiselevel,
      NULL));
  PetscOptionsEnd();

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, 900, 2, 1, NULL, &da));
  //  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
  //               DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
  //               DMDA_STENCIL_BOX,  // for 9-point stencil
  //               3,3,PETSC_DECIDE,PETSC_DECIDE,
  //               2, 1,              // degrees of freedom, stencil width
  //               NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da, 0, "a"));
  PetscCall(DMDASetFieldName(da, 1, "q"));
  PetscCall(DMDAGetLocalInfo(da, &info));

  // PetscCall(DMDASetUniformCoordinates(da, 0.0, user.L));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "running on %d grid with cells of size h = %.6f ...\n",
      info.mx,
      user.L / (PetscReal)(info.mx)));

  // STARTTSSETUP
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetApplicationContext(ts, &user));
  PetscCall(DMDATSSetRHSFunctionLocal(
      da, INSERT_VALUES, (DMDATSRHSFunctionLocal)FormRHSFunctionLocal, &user));
  if (!no_rhsjacobian) {
    PetscCall(DMDATSSetRHSJacobianLocal(da, (DMDATSRHSJacobianLocal)FormRHSJacobianLocal, &user));
  }
  PetscCall(
      DMDATSSetIFunctionLocal(da, INSERT_VALUES, (DMDATSIFunctionLocal)FormIFunctionLocal, &user));
  if (!no_ijacobian) {
    PetscCall(DMDATSSetIJacobianLocal(da, (DMDATSIJacobianLocal)FormIJacobianLocal, &user));
  }
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetTimeStep(ts, 0.1));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));
  // ENDTSSETUP

  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(InitialState(da, x, noiselevel, &user));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial State\n"));

  PetscCall(TSSolve(ts, x));

  // optionally report on call-backs
  if (call_back_report) {
    PetscCall(TSGetType(ts, &type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CALL-BACK REPORT\n  solver type: %s\n", type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  IFunction:   %d  | IJacobian:   %d\n",
        (int)user.IFcn_called,
        (int)user.IJac_called));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  RHSFunction: %d  | RHSJacobian: %d\n",
        (int)user.RHSFcn_called,
        (int)user.RHSJac_called));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

// Find in Oscar book or other a good initial state a_o(x) and q_o(x)
PetscErrorCode InitialState(DM da, Vec Y, PetscReal noiselevel, PatternCtx* user)
{
  DMDALocalInfo info;
  PetscInt i;
  PetscReal sx;
  const PetscReal ledge = (user->L - 0.5) / 2.0,  // nontrivial initial values on
      redge = user->L - ledge;  //   ledge < x,y < redge
  Vec aC;
  Field* aY;

  PetscCall(VecSet(Y, 0.0));
  if (noiselevel > 0.0) {
    // noise added to usual initial condition is uniform on [0,noiselevel],
    //     independently for each location and component
    PetscCall(VecSetRandom(Y, NULL));
    PetscCall(VecScale(Y, noiselevel));
  }
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMGetCoordinates(da, &aC));
  PetscCall(DMDAVecGetArray(da, Y, &aY));

  for (i = info.xs; i < info.xs + info.xm; i++) {
    aY[i].q = 10.0;
    aY[i].a = 20.0;
  }
  PetscCall(DMDAVecRestoreArray(da, Y, &aY));
  // PetscCall(DMDAVecRestoreArray(da,aC, &aC));
  return 0;
}

// in system form  F(t,Y,dot Y) = G(t,Y),  compute G():
//     G^a(t,a,q) = 0
//     G^q(t,a,q) = g a (So-Sf)
//     So = - dz/dx
//     Sf = nm^2 q |q| a^{-10/3}
// STARTRHSFUNCTION
PetscErrorCode FormRHSFunctionLocal(
    DMDALocalInfo* info, PetscReal t, Field* aY, Field* aG, PatternCtx* user)
{
  PetscInt i;
  PetscReal Sf;

  user->RHSFcn_called = PETSC_TRUE;
  for (i = info->xs; i < info->xs + info->xm; i++) {
    Sf = (user->nm) * (user->nm) * aY[i].q * PetscAbsReal(aY[i].q)
        * PetscPowReal(aY[i].a, -10.0 / 3.0);
    aG[i].a = 0.0;
    aG[i].q = (user->g) * aY[i].a * (user->So - Sf);
  }
  return 0;
}

// ENDRHSFUNCTION

PetscErrorCode FormRHSJacobianLocal(
    DMDALocalInfo* info, PetscReal t, Field* aY, Mat J, Mat P, PatternCtx* user)
{
  PetscInt i;
  PetscReal v[2], Sf, SfDa, SfDq;
  MatStencil col[2], row;

  user->RHSJac_called = PETSC_TRUE;
  for (i = info->xs; i < info->xs + info->xm; i++) {
    row.i = i;
    col[0].i = i;
    col[1].i = i;
    Sf = (user->nm) * (user->nm) * aY[i].q * PetscAbsReal(aY[i].q)
        * PetscPowReal(aY[i].a, -10.0 / 3.0);
    SfDa = -10.0 / 3.0 * (user->nm) * (user->nm) * aY[i].q * PetscAbsReal(aY[i].q)
        * PetscPowReal(aY[i].a, -13.0 / 3.0);
    SfDq = (user->nm) * (user->nm) * PetscAbsReal(aY[i].q) * PetscPowReal(aY[i].a, -10.0 / 3.0)
        + (user->nm) * (user->nm) * aY[i].q * PetscCopysignReal(1.0, aY[i].q)
            * PetscPowReal(aY[i].a, -10.0 / 3.0);
    // a equation
    row.c = 0;
    col[0].c = 0;
    col[1].c = 1;
    v[0] = 0.0;
    v[1] = 0.0;
    PetscCall(MatSetValuesStencil(P, 1, &row, 2, col, v, INSERT_VALUES));

    // q equation
    row.c = 1;
    col[0].c = 0;
    col[1].c = 1;
    v[0] = (user->g) * (user->So - Sf) - (user->g) * aY[i].a * SfDa;
    v[1] = -(user->g) * aY[i].a * SfDq;
    PetscCall(MatSetValuesStencil(P, 1, &row, 2, col, v, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  return 0;
}

// in system form  F(t,Y,dot Y) = G(t,Y),  compute F():
//     F^u(t,a,q,a_t,q_t) = a_t +  Dx[q]
//     F^v(t,a,q,a_t,q_t) = q_t +  Dx[q^2/a+ga^2/2] = q_t + 2q Dx[q]/a-q^2/a^2 Dx[a]+g a Dx[a]
// STARTIFUNCTION
PetscErrorCode FormIFunctionLocal(
    DMDALocalInfo* info, PetscReal t, Field* aY, Field* aYdot, Field* aF, PatternCtx* user)
{
  PetscInt i, j;
  const PetscReal h = user->L / (PetscReal)(info->mx), C = 1.0 / (2.0 * h);
  PetscReal a, q, daf, dqf;

  user->IFcn_called = PETSC_TRUE;
  i = info->xs - 1;
  aY[i].a = aY[i + 1].a;
  aY[i].q = aY[i + 1].q;
  i = info->xs + info->xm;
  aY[i].a = aY[i - 1].a;
  aY[i].q = aY[i - 1].q;
  for (i = info->xs; i < info->xs + info->xm; i++) {
    a = aY[i].a;
    q = aY[i].q;
    daf = (aY[i + 1].a - aY[i - 1].a);
    dqf = (aY[i + 1].q - aY[i - 1].q);
    aF[i].a = aYdot[i].a + C * dqf;
    aF[i].q = aYdot[i].q + C * (2.0 * q * dqf / a - q * q * daf / (a * a) + (user->g) * a * daf);

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
}

// ENDIFUNCTION

// in system form  F(t,Y,dot Y) = G(t,Y),  compute combined/shifted
// Jacobian of F():
//     J = (shift) dF/d(dot Y) + dF/dY
// STARTIJACOBIAN
PetscErrorCode FormIJacobianLocal(DMDALocalInfo* info,
    PetscReal t,
    Field* aY,
    Field* aYdot,
    PetscReal shift,
    Mat J,
    Mat P,
    PatternCtx* user)
{
  PetscInt i, s, cr, cc;
  const PetscReal h = user->L / (PetscReal)(info->mx), C = 1.0 / (2.0 * h);
  PetscReal val[3], CC, a, q, daf, dqf;
  MatStencil col[3], row;

  PetscCall(MatZeroEntries(P));  // workaround to address PETSc issue #734
  user->IJac_called = PETSC_TRUE;
  i = info->xs - 1;
  aY[i].a = aY[i + 1].a;
  aY[i].q = aY[i + 1].q;
  i = info->xs + info->xm;
  aY[i].a = aY[i - 1].a;
  aY[i].q = aY[i - 1].q;
  for (i = info->xs; i < info->xs + info->xm; i++) {
    row.i = i;

    a = aY[i].a;
    q = aY[i].q;
    daf = (aY[i + 1].a - aY[i - 1].a);
    dqf = (aY[i + 1].q - aY[i - 1].q);
    cr = 0;
    row.c = cr;

    //          aF[i].a = aYdot[i].a+C*dqf;
    cc = 0;
    col[0].c = cc;
    col[0].i = i - 1;
    val[0] = 0.0;
    col[1].c = cc;
    col[1].i = i;
    val[1] = shift;
    col[2].c = cc;
    col[2].i = i + 1;
    val[2] = 0.0;
    PetscCall(MatSetValuesStencil(P, 1, &row, 3, col, val, INSERT_VALUES));

    cc = 1;
    col[0].c = cc;
    col[0].i = i - 1;
    val[0] = -C;
    col[1].c = cc;
    col[1].i = i;
    val[1] = 0.0;
    col[2].c = cc;
    col[2].i = i + 1;
    val[2] = C;
    PetscCall(MatSetValuesStencil(P, 1, &row, 3, col, val, INSERT_VALUES));

    //          aF[i].q = aYdot[i].q+C*(2.0*q*dqf/a-q*q*daf/(a*a)+ (user->g)*a*daf);
    cr = 1;
    row.c = cr;
    cc = 0;
    col[0].c = cc;
    col[0].i = i - 1;
    val[0] = C * (q * q / (a * a) - (user->g) * a);
    col[1].c = cc;
    col[1].i = i;
    val[1] = C * (-2.0 * q * dqf / (a * a) + 2.0 * q * q * daf / (a * a * a) + (user->g) * daf);
    col[2].c = cc;
    col[2].i = i + 1;
    val[2] = C * (-q * q / (a * a) + (user->g) * a);
    PetscCall(MatSetValuesStencil(P, 1, &row, 3, col, val, INSERT_VALUES));

    cc = 1;
    col[0].c = cc;
    col[0].i = i - 1;
    val[0] = -C * 2.0 * q / a;
    col[1].c = cc;
    col[1].i = i;
    val[1] = shift + C * (2.0 * dqf / a - 2 * q * daf / (a * a));
    col[2].c = cc;
    col[2].i = i + 1;
    val[2] = C * 2.0 * q / a;
    PetscCall(MatSetValuesStencil(P, 1, &row, 3, col, val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  return 0;
}

// ENDIJACOBIAN
