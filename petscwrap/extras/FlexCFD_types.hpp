template<typename funname, typename... xargs>
using spacefun = funname (*)(xargs...);

using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

namespace FCFD
{
namespace PDEModel
{

enum class PDEType
{
  Unknown,
  SVE,
  SWE2d,
  SWE3d
};

// A modular grid parameters struct using a non-type template parameter for the dimension.
// This structure stores domain lengths and grid spacings in contiguous arrays.
template<size_t Dim>
struct grid_params_t
{
  PDEType type;
  std::array<double, Dim> side_lengths;  // Domain lengths in each dimension.
  std::array<double, Dim> spacings;  // Grid spacings (e.g., dx, dy, dz).
  double TotalT;
  double dt;
};

// Aliases for common grid dimensions.
using grid_1d_params = grid_params_t<1>;
using grid_2d_params = grid_params_t<2>;
using grid_3d_params = grid_params_t<3>;

template<typename nativet>
struct model_params
{
  constexpr static nativet default_g = (nativet)9.81;
  nativet g, nu, gammat, gamman, nm, wsurf, radi;
};

template<typename nativet>
struct model_2d_params : model_params<nativet>
{
  nativet wsurfydir;
};

template<typename pdeopts>
struct PDEParams
{
  pdeopts mypdeopts;
  int nd, nv, nbound;

  PDEParams(int nd_, int nv_);
};

// BoundaryConditions
template<typename nativet>
struct BoundaryCondition
{
  std::function<std::vector<nativet>(std::span<const nativet>)> cond_bound_funs;
  int bd_type;  // e.g., 1: Dirichlet, 2: Neumann, 3: Robin.
  std::function<std::vector<nativet>(std::span<const nativet>)> bdryfunop;
  std::function<std::vector<nativet>(std::span<const nativet>)> id_to_bdryfunrhs;
};

// PDESystem is now templated on nativet so that the same class can work with different native
// types.
template<typename nativet, typename pdeopts>
class PDESystem
{
public:
  using SystemFunctionType = std::function<std::vector<nativet>(std::vector<nativet>)>;
  // Virtual destructor for proper cleanup.
  virtual ~PDESystem() = default;

  // Pure virtual function for SetBottom (if needed) plus an overload.
  virtual void SetBottom() = 0;

  void SetBottom(const std::function<std::vector<nativet>(std::span<const nativet>)>& seafloor)
  {
    groundfun = seafloor;
    BottomSet = true;
  }

  void InitPDESystem(pdeopts options)
  {
    mypdeopts = options;
  };

  // Set the initial condition function and mark it as set.
  void SetInitCond(const std::function<std::vector<nativet>(std::vector<nativet>)>& initconds)
  {
    MyInitconds = initconds;
    InitCondSet = true;
  }

  virtual void SetInitCond() = 0;

  // Set boundary conditions.
  void SetBdryCond(std::map<int, BoundaryCondition<nativet>>&& bdryconds, int numconds)
  {
    mybdry = std::move(bdryconds);
    nbound = numconds;
    BdryCondSet = true;
  }

  virtual void SetBdryCond() = 0;

  // RHS now takes an input vector and returns an output vector.
  void RHS(const std::vector<nativet>* x, std::vector<nativet>& out)
  {
    assert(BoolRHSSet && "RHS function has not been set!");
    out = MyRHS(x);
  }

  // Jacobian RHS operator.
  void JacRHS(const std::vector<nativet>* x, std::vector<nativet>& out, int ro)
  {
    assert(BoolJacRHSSet && "JacRHS function has not been set!");
    out = MyJacRHS(x, ro);
  }

  // LHS operator.
  void LHSOpAsplit(const std::vector<nativet>* v,
      const std::vector<nativet>* vdot,
      const std::vector<nativet>* dvdx,
      const std::vector<nativet>* dv2dx,
      const std::vector<nativet>* dv3dx,
      std::vector<nativet>& out)
  {
    assert(BoolLHSOpSet && "LHSOp function has not been set!");
    out = MyLHSAsplit(v, vdot, dvdx, dv2dx, dv3dx);
  }

  void LHSOpBsplit(const std::vector<nativet>* v,
      const std::vector<nativet>* vdot,
      const std::vector<nativet>* dvdx,
      const std::vector<nativet>* dv2dx,
      const std::vector<nativet>* dv3dx,
      std::vector<nativet>& out)
  {
    assert(BoolLHSOpSet && "LHSOp function has not been set!");
    out = MyLHSBsplit(v, vdot, dvdx, dv2dx, dv3dx);
  }

  // Jacobian LHS operator.
  void JacLHSOpAsplit(const std::vector<nativet>* v,
      const std::vector<nativet>* vdot,
      const std::vector<nativet>* dvdx,
      const std::vector<nativet>* dv2dx,
      const std::vector<nativet>* dv3dx,
      std::vector<nativet>* out,
      int rowo,
      int colo,
      int derivo)
  {
    assert(BoolJacLHSOpSet && "JacLHSOp function has not been set!");
    out = MyJacLHSAsplit(v, vdot, dvdx, dv2dx, dv3dx, rowo, colo, derivo);
  }

  void JacLHSOpBsplit(const std::vector<nativet>& v,
      const std::vector<nativet>& vdot,
      const std::vector<nativet>& dvdx,
      const std::vector<nativet>& dv2dx,
      const std::vector<nativet>& dv3dx,
      std::vector<nativet>& out,
      int rowo,
      int colo,
      int derivo)
  {
    assert(BoolJacLHSOpSet && "JacLHSOp function has not been set!");
    out = MyJacLHSBsplit(v, vdot, dvdx, dv2dx, dv3dx, rowo, colo, derivo);
  }

  // Pure virtual functions for initial and boundary condition evaluations.
  virtual nativet InitCond() = 0;
  virtual nativet BdryCond() = 0;

  // Setter for the RHS function.
  void SetRHS(const std::function<std::vector<nativet>(std::vector<nativet>)>& rhsconds)
  {
    MyRHS = rhsconds;
    BoolRHSSet = true;
  }

  virtual void SetRHS() = 0;

  // Setter for the Jacobian RHS function.
  void SetJacRHS(const std::function<std::vector<nativet>(std::vector<nativet>, int)>& jacrhsconds)
  {
    MyJacRHS = jacrhsconds;
    BoolJacRHSSet = true;
  }

  virtual void SetJacRHS() = 0;

  // Setter for the LHS operator function. A/B split for implicit/explicit distinguishing
  void SetLHSOp(const std::function<std::vector<nativet>(std::vector<nativet>*,
                    std::vector<nativet>*,
                    std::vector<nativet>*,
                    std::vector<nativet>*,
                    std::vector<nativet>*)>& lhsa,
      const std::function<std::vector<nativet>(std::vector<nativet>*,
          std::vector<nativet>*,
          std::vector<nativet>*,
          std::vector<nativet>*,
          std::vector<nativet>*)>& lhsb)
  {
    MyLHSAsplit = lhsa;
    MyLHSBsplit = lhsb;

    BoolLHSOpSet = true;
  }

  virtual void SetLHSOp() = 0;

  // Setter for the Jacobian LHS operator function.

  void SetJacLHSOp(const std::function<std::vector<nativet>(std::vector<nativet>*,
                       std::vector<nativet>*,
                       std::vector<nativet>*,
                       std::vector<nativet>*,
                       std::vector<nativet>*,
                       int,
                       int,
                       int)>& jaclhsa,
      const std::function<std::vector<nativet>(std::vector<nativet>*,
          std::vector<nativet>*,
          std::vector<nativet>*,
          std::vector<nativet>*,
          std::vector<nativet>*,
          int,
          int,
          int)>& jaclhsb)
  {
    MyJacLHSAsplit = jaclhsa;
    MyJacLHSBsplit = jaclhsb;
    BoolJacLHSOpSet = true;
  }

  virtual void SetJacLHSOp() = 0;

  void InitPDESys();

  // EvalSol calls the private 'Solution' function.
  void EvalSol(const std::vector<nativet>& x, std::vector<nativet>& out)
  {
    assert(SolFound && "Solution function has not been set!");
    out = Solution(x);
  }

protected:
  PDESystem(PDEType pdetype_, PDEParams<pdeopts>& params_)
      : pdetype(pdetype_)
      , mypdeparams(params_)
  {
  }

  bool BottomSet = false;
  bool InitCondSet = false;
  bool BdryCondSet = false;
  bool paramsset = false;
  bool gridparamsset = false;

  bool BoolRHSSet = false;
  bool BoolJacRHSSet = false;
  bool BoolLHSOpSet = false;
  bool BoolJacLHSOpSet = false;
  bool SolFound = false;

  SystemFunctionType groundfun;

  // Note: pdetype and nd are declared only once.
  PDEType pdetype;
  pdeopts mypdeopts;
  PDEParams<pdeopts> mypdeparams;
  int nd, nv, nbound;

  std::map<int, BoundaryCondition<nativet>> mybdry;
  const std::map<int, BoundaryCondition<nativet>> default_boundary;
  SystemFunctionType default_initconds;
  SystemFunctionType MyInitconds;
  SystemFunctionType MyLHSAsplit;
  SystemFunctionType MyJacLHSAsplit;
  SystemFunctionType MyLHSBsplit;
  SystemFunctionType MyJacLHSBsplit;
  SystemFunctionType MyRHS;
  SystemFunctionType MyJacRHS;
  SystemFunctionType Solution;
};

template<typename nativet, typename pdeopts>
class SVE : public PDESystem<nativet, pdeopts>
{
public:
  using PDESystem<nativet, pdeopts>::InitPDESys;

  SVE()
      : PDESystem<nativet, pdeopts>(PDEType::SVE, PDEParams(1, 2, 0, pdeopts()))
      , mysveparams(default_model_params)
      , my_grid_params(default_grid_params)

  {
    InitPDESys();
  };

  SVE(model_params params_in, int nfield, grid_1d_params grid_params)
      : PDESystem<nativet, pdeopts>(PDEType::SVE, PDEParams(1, nfield, 0, pdeopts()))
      , mysveparams(params_in)
      , my_grid_params(grid_params)
  {
    InitPDESys();
  };

  grid_1d_params get_my_params()
  {
    return mysveparams;
  }

  grid_1d_params get_my_grid_params()
  {
    return my_grid_params;
  }

private:
  model_params mysveparams;
  static inline const model_params default_model_params = {9.8, 1.0, 0.1, 0.8, 1.0, 2.0, 0.7};
  grid_1d_params my_grid_params;
  static inline const grid_1d_params default_grid_params = {PDEType::SVE, {1.}, {0.01}, 10, 0.01};
};

template<typename nativet, typename pdeopts>
class SWE2d : public PDESystem<nativet, pdeopts>
{
public:
  // Empty virtual destructor for proper cleanup
  virtual ~SWE2d() = 0;

  using PDESystem<nativet, pdeopts>::InitPDESys;

  SWE2d()
      : PDESystem<nativet, pdeopts>(PDEType::SWE2d, PDEParams(2, 3, 0, pdeopts()))
      , mysweparams(default_model_params)
      , my_grid_params(default_grid_params)
  {
    InitPDESys();
  };

  SWE2d(model_params params_in, int nfield, grid_2d_params grid_params)
      : PDESystem<nativet, pdeopts>(PDEType::SWE2d, PDEParams(2, nfield, 0, pdeopts()))
      , mysweparams(params_in)
      , my_grid_params(grid_params)
  {
    InitPDESys();
  };

private:
  model_params mysweparams;
  static inline const model_2d_params default_model_params = {
      9.8, 1.0, 0.1, 0.8, 1.0, 2.0, 0.7, 1.5};
  grid_2d_params my_grid_params;
  static inline const grid_2d_params default_grid_params = {
      PDEType::SWE2d, {1., 0.1}, {0.01, 0.001}, 10, 0.01};
};

};  // namespace PDEModel

namespace PDENumerics
{
template<typename nativet, typename pdeopts, typename errtype, typename methodp, typename optform>
class SolTool
{
public:
  virtual errtype InitSolMethod();

  void InitSolMethod(std::unique_ptr<methodp>&& solminit)
  {
    methodprops = std::move(solminit);
    setInit[0] = true;
  };

  virtual errtype InitProblem();
  virtual errtype SetSolOpts();

  void SetSolOpts(std::unique_ptr<optform> opts)
  {
    algopts = std::move(opts);
    setInitFlag(1, true);
  };

  virtual errtype SetPDE();

  void setPDE(std::unique_ptr<PDESystem<nativet, pdeopts>> initPDE)
  {
    myPDE = std::move(initPDE);
  }

  virtual errtype NumericalSolve();
  virtual errtype InterpolateSol(
      std::function<std::vector<double>(std::vector<nativet>)>& interpol);

protected:
  void setInitFlag(size_t idx, bool val)
  {
    initialized[idx] = val;
  }

  bool getInitFlag(size_t idx) const
  {
    .return initialized[idx];
  }

  std::unique_ptr<PDESystem<nativet, pdeopts>> myPDE;
  std::array<bool, 3> initialized = {false, false, false, false};
  methodp methodprops;
  probform methodform;
  optform algopts;
  int nd;
};

struct PetscOpts
{
  TSType petopt_tstype = {};
  int nfield = 0;
  std::vector<string> fields;
};

/*/       struct PetscProblem {
         int bdycon = 0;
         int setcount = 0;
         std::vector<int> stendata;
         std::function<PetscErrorCode(DM, Vec)> InitialState;
         std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**, Vec)>
   FormRHSFunctionLocal; std::function<PetscErrorCode(PetscReal, PetscReal**, Mat, Mat)>
   FormRHSJacobianLocal; std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**,
   PetscReal**)> FormIFunctionLocal; std::function<PetscErrorCode(PetscReal, PetscReal**,
   PetscReal**, PetscReal, Mat, Mat)> FormIJacobianLocal;

      };*/
struct PetscSolveOpts
{
  TSType mytstype = TSARKIMEX;
  PetscReal inittime = 0.0;
  PetscReal maxtime = 10.0;
  PetscReal timestep = 0.1;
  TSExactFinalTimeOption fintimeopt = TS_EXACTFINALTIME_MATCHSTEP;
};

/* this stuff:
PetscCall(TSSetType(ts,TSARKIMEX));
PetscCall(TSSetTime(ts,0.0));
PetscCall(TSSetMaxTime(ts,10.0));
PetscCall(TSSetTimeStep(ts,0.1));
PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
PetscCall(TSSetFromOptions(ts));
*/

class PetscInitException
{
};

inline constexpr bool IsPetscError(PetscErrorCode code)
{
  return code != PETSC_SUCCESS;
}

struct CallBacks
{
  PetscBool IFcn_called, IJac_called, RHSFcn_called, RHSJac_called;
};

template<typename pdeopts>
class PetscWrap : public SolTool<PetscReal, pdeopts, PetscErrorCode, PetscOpts, PetscSolveOpts>
{
private:
  using BaseClass = SolTool<PetscReal, pdeopts, PetscErrorCode, PetscOpts, PetscSolveOpts>;
  using BaseClass::getInitFlag;
  using BaseClass::setInitFlag;

  PetscOpts methodprops;
  PetscSolveOpts algopts;
  CallBacks user;

  TS MyTS;
  DM da;
  DMDALocalInfo info;
  TSType type;
  Vec x;
  PetscBool no_rhsjacobian {PETSC_FALSE};
  PetscBool no_ijacobian {PETSC_FALSE};
  PetscBool call_back_report {PETSC_TRUE};
  std::vector<int> mystendata;
  std::optional<std::vector<PetsReal>> mystengridd;
  std::optional<std::vector<PetsReal>> mystengriddd;
  std::optional<std::vector<PetsReal>> mystengridd3;
  std::vector<PetscReal> intmp;
  std::vector<PetscReal> outtmp;
  std::vector<PetscReal> vtmp;
  std::vector<PetscReal> vtmptmp;

  std::vector<MatStencil> coltmp;
  MatStencil rowtmp;
  std::vector<PetscReal> indtmp;
  std::vector<PetscReal> outdtmp;
  std::vector<PetscReal> vdtmp;
  std::vector<MatStencil> coldtmp;
  MatStencil rowdtmp;
  std::optional < std::vector < PetscReal >>> ind2tmp;
  std::optional<std::vector<PetscReal>> outd2tmp;
  std::optional<std::vector<PetscReal>> vd2tmp;
  std::optional<std::vector<MatStencil>> cold2tmp;
  std::optional<MatStencil> rowd2tmp;
  std::optional < std::vector < PetscReal >>> ind3tmp;
  std::optional<std::vector<PetscReal>> outd3tmp;
  std::optional<std::vector<PetscReal>> vd3tmp;
  std::optional<std::vector<MatStencil>> cold3tmp;
  std::optional<MatStencil> rowd3tmp;
  /*         std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**, Vec)>
     FormRHSFunctionLocal; std::function<PetscErrorCode(PetscReal, PetscReal**, Mat, Mat)>
     FormRHSJacobianLocal; std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**,
     PetscReal**)> FormIFunctionLocal; std::function<PetscErrorCode(PetscReal, PetscReal**,
     PetscReal**, PetscReal, Mat, Mat)> FormIJacobianLocal;
  */
  static PetscErrorCode FormRHSFunctionLocal(
      DMDALocalInfo* info, PetscReal t, (PestcReal*)aY, (PestcReal*)aG, void* pwp);
  // ENDRHSFUNCTION

  static PetscErrorCode FormRHSJacobianLocal(
      DMDALocalInfo* info, PetscReal t, (PestcReal*)aY, Mat J, Mat P, void* pwp);

  // in system form  F(t,Y,dot Y) = G(t,Y),  compute F():
  //     F^u(t,a,q,a_t,q_t) = a_t +  Dx[q]
  //     F^v(t,a,q,a_t,q_t) = q_t +  Dx[q^2/a+ga^2/2] = q_t + 2q Dx[q]/a-q^2/a^2 Dx[a]+g a Dx[a]
  // STARTIFUNCTION
  static PetscErrorCode FormIFunctionLocal(DMDALocalInfo* info,
      PetscReal t,
      (PestcReal*)aY,
      (PestcReal*)aYdot,
      (PestcReal*)aF,
      void* pwp);
  // ENDIFUNCTION

  // in system form  F(t,Y,dot Y) = G(t,Y),  compute combined/shifted
  // Jacobian of F():
  //     J = (shift) dF/d(dot Y) + dF/dY
  // STARTIJACOBIAN
  static PetscErrorCode FormIJacobianLocal(DMDALocalInfo* info,
      PetscReal t,
      (PestcReal*)aY,
      (PestcReal*)aYdot,
      PetscReal shift,
      Mat J,
      Mat P,
      void* pwp);

public:
  PetscWrap(int argc, char** argv, const char* msg)
  {
    if (IsPetscError(PetscInitialize(&argc, &argv, NULL, msg))) {
      throw PetscInitException();
    }
    user.IFcn_called = PETSC_FALSE;
    user.IJac_called = PETSC_FALSE;
    user.RHSFcn_called = PETSC_FALSE;
    user.RHSJac_called = PETSC_FALSE;
               if (IsPetscError(MakeProbFuns())
               {
      throw PetscInitException();
               }
  };

  PetscWrap(int numf)
      : PetscWrap()
  {
    methodprops.nfield = numf;
    setInitFlag(0, true);
  };

  PetscWrap(int argc, char** argv, const char* msg, int numf)
      : PetscWrap(argc, argv, msg)
  {
    methodprops.nfield = numf;
    setInitFlag(0, true);
  };

  PetscWrap(int argc, char** argv, const char* msg, TSType tst, int numf)
      : PetscWrap(argc, argv, msg)
  {
    methodprops.nfield = numf;
    methodprops.petopt_tstype = tst;
    setInitFlag(0, true);
  };

  PetscWrap(int argc, char** argv, const char* msg, TSType tst, std::vector<string> FieldNames)
      : PetscWrap(argc, argv, msg)
  {
    methodprops.petopt_tstype = tst;
    this->SetFields(FieldNames);
    setInitFlag(0, true);
  };

  PetscWrap(int argc, char** argv, const char* msg, std::vector<string> FieldNames)
      : PetscWrap(argc, argv, msg)
  {
    this->SetFields(FieldNames);
    setInitFlag(0, true);
  };

  PetscWrap();

  PetscWrap(std::vector<string> FieldNames)
  {
    PetscWrap();
    this->SetFields(FieldNames);
    setInitFlag(0, true);
  };

  ~PetscWrap()
  {
    if (&x) {
      VecDestroy(&x);
    }
    if (&MyTS) {
      TSDestroy(&MyTS);
    }
    if (&da) {
      DMDestroy(&da);
    }
             if (IsPetscError(PetscFinalize())
               {
      // throw PetscInitException();
               }
  }

  virtual void SetFields();
  PetscErrorCode SetFields(std::vector<string> FieldNames);
  PetscErrorCode InitSolMethod() override;
  PetscErrorCode InitProblem();

  PetscErrorCode InitProblem(const PetscProblem& probinit);
  virtual void SetGrid();
  PetscErrorCode SetGrid(int ndim,
      std::vec<int> bdycon,
      std::vector<int> stendata,
      std::vector<PetscReal>* stengrid,
      std::vector<PetscReal>* stengridd,
      std::vector<PetscReal>* stengridd3);

  /*          void SetInitState(std::function<PetscErrorCode(DM,Vec)> initstate) {

              methodform.InitialState = std::move(initstate);
              if (methodform.setcount < 5) methodform.setcount++;
              if (methodform.setcount == 5) setInitFlag(1,true);

            }*/
  /*          void InitProblembyPDE() {
               methodform.FormRHSFunctionLocal = FormRHSbyPDE;
               methodform.FormRHSJacobianLocal = FormRHSJacbyPDE;
               methodform.FormIFunctionLocal = FormLHSbyPDE;
               methodform.FormIJacobianLocal = FormLHSJacbyPDE;
               if (methodform.setcount < 5) methodform.setcount = methdform.setcount+4;
               if (methodform.setcount == 5) setInitFlag(1,true);

            }
            void InitProblem(std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**,
     Vec)>&& FormRHSInit, std::function<PetscErrorCode(PetscReal, PetscReal**, Mat, Mat)>&&
     FormRHSJacInit, std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**,
     PetscReal**)>&& FormLHSInit, std::function<PetscErrorCode(PetscReal, PetscReal**, PetscReal**,
     PetscReal, Mat, Mat)>&& FormLHSJacInit) {

              methodform.FormRHSFunctionLocal = std::move(FormRHSInit);
              methodform.FormRHSJacobianLocal = std::move(FormRHSJacInit);
              methodform.FormIFunctionLocal = std::move(FormLHSInit);
              methodform.FormIJacobianLocal = std::move(FormLHSJacInit);
              if (methodform.setcount < 5) methodform.setcount = methodform.setcount + 4;
              if (methodform.setcount == 5) setInitFlag(1,true);


            }*/
  PetscErrorCode MakeProbFuns()
  {
    PetscCall(DMDATSSetRHSFunctionLocal(
        da, INSERT_VALUES, (DMDATSRHSFunctionLocal)FormRHSFunctionLocal, this));
    PetscCall(DMDATSSetRHSJacobianLocal(da, (DMDATSRHSJacobianLocal)FormRHSJacobianLocal, this));
    PetscCall(
        DMDATSSetIFunctionLocal(da, INSERT_VALUES, (DMDATSIFunctionLocal)FormIFunctionLocal, this));
    PetscCall(DMDATSSetIJacobianLocal(da, (DMDATSIJacobianLocal)FormIJacobianLocal, this));
  }

  PetscErrorCode SetSolOpts()
  {
    if (getInitFlag(3)) {
      PetscCall(TSSetType(MyTS, algopts.mytstype));
      PetscCall(TSSetTime(MyTS, algopts.inittime));
      PetscCall(TSSetMaxTime(MyTS, algopts.maxtime));
      PetscCall(TSSetTimeStep(MyTS, algopts.timestep));
      PetscCall(TSSetExactFinalTime(MyTS, algopts.fintimeopt));
      PetscCall(TSSetFromOptions(MyTS));
    }
  }

  void SetSolOpts(const PetscSolveOpts& opts)
  {
    algopts = opts;
    setInitFlag(3, true);
    this->SetSolOpts();
  };

  void SetTSScheme(ts setts)
  {
    myTS = std::move(setts);
  }

  PetscErrorCode NumericalSolve()
  {
    PetscCall(DMCreateGlobalVector(da, &x));
    PetscCall(InitialState(da, x));
    PetscCall(TSSolve(MyTS, x));
  }

  virtual PetscErrorCode InterpolateSol(
      std::function<std::vector<double>(std::vector<PetscReal>)>& interpol);
  // write implementation

private:
};

/*template<typename nativet, typename errtype, typename methodp, typename probform, typename
   optform> class DfxWrap : public SolTool<nativet, errtype, methodp, probform, optform> { public:
         virtual void SetMesh();
         virtual void SetFuncSpaces();
         virtual void SetForms();
         virtual void InitTimeStepping();
         virtual void GoTimeStep();

       protected:

     };*/
}  // namespace PDENumerics
}  // namespace FCFD
