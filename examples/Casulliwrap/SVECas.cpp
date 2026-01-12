// Example/main_1d.cpp
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "numerics/casulliwrap/CasulliWrap.hpp"
#include "pdes/sve/SVECas.hpp"

struct MyPDEOptions
{
};

namespace
{
using Real = double;

static std::string get_arg(int argc, char** argv, int k, std::string fallback = "")
{
  return (argc > k && argv[k]) ? std::string(argv[k]) : fallback;
}

static std::size_t clamp_stride(std::size_t n, std::size_t target_points = 12)
{
  if (n == 0) return 1;
  std::size_t stride = n / target_points;
  return std::max<std::size_t>(1, stride);
}

template<class T>
static void print_1d_sample(
  const std::string& name, const std::vector<T>& v, std::size_t stride, std::size_t max_rows = 200)
{
  std::cout << "\n" << name << " (sampled), n=" << v.size() << ", stride=" << stride << "\n";
  std::size_t rows = 0;
  for (std::size_t i = 0; i < v.size(); i += stride) {
    std::cout << name << "[" << i << "]=" << std::setprecision(16) << v[i] << "\n";
    if (++rows >= max_rows) break;
  }
  if (!v.empty() && ((v.size() - 1) % stride != 0) && rows < max_rows) {
    std::cout << name << "[" << (v.size() - 1) << "]=" << std::setprecision(16) << v.back() << "\n";
  }
}

static void print_usage(char** argv)
{
  std::cout << "Usage:\n"
            << "  " << argv[0] << " N [ic]\n\n"
            << "Where:\n"
            << "  N   : grid size (1D: N cells)\n"
            << "  ic  : initial condition selector (optional)\n"
            << "       uniform  -> use PDE SetIC() values (uniform)\n"
            << "       step     -> 1D dam-break step (default)\n\n"
            << "Examples:\n"
            << "  " << argv[0] << " 41\n"
            << "  " << argv[0] << " 41 uniform\n";
}
}  // namespace

int main(int argc, char** argv)
{
  const std::string a1 = get_arg(argc, argv, 1, "");
  if (a1 == "--help" || a1 == "-h") {
    print_usage(argv);
    return 0;
  }

  const std::size_t N = (argc > 1) ? static_cast<std::size_t>(std::stoul(argv[1])) : 41;
  const std::string ic_mode = get_arg(argc, argv, 2, "step");
  const bool use_uniform_ic = (ic_mode == "uniform");

  std::cout << "=== 1D CASE (SVEcas) ===\n";
  std::cout << "N=" << N << "  ic=" << ic_mode << "\n";

  auto pde = std::make_unique<fcfd::pdemodel::SVEcas<Real, MyPDEOptions>>();

  // Model params
  pde->GetModelParams().g = 9.81;
  pde->GetModelParams().nm = 0.03;
  pde->GetModelParams().radi = 1.0;
  pde->GetModelParams().gammat = 0.1;

  // Uniform IC (used if ic=uniform)
  pde->SetIC(0.0, 0.0);  // (eta0, u0)
  pde->SetBed(0.0, 0.0);
  pde->SetWindSpeedU(0.0);

  // Example BCs
  pde->SetLeftEta(0.0);
  pde->SetRightU(0.0);

  fcfd::pdenumerics::CasulliWrap<Real, MyPDEOptions> wrap;
  wrap.SetGrid1D(N);

  wrap.MethodProps().dt = 0.1;
  wrap.MethodProps().dx = 1.0;
  wrap.MethodProps().nSteps = 5;
  wrap.MethodProps().dryTol = 1e-12;
  wrap.MethodProps().newtonMaxIts = 20;
  wrap.MethodProps().newtonTol = 1e-10;
  wrap.SetNewtonVerbose(true);

  if (!use_uniform_ic) {
    wrap.SetInitialEta1D(
      [&](std::size_t, double x)
      {
        const double xmid = 0.5 * (N - 1) * wrap.MethodProps().dx;
        return (x < xmid) ? 0.2 : 0.0;
      });
    wrap.SetInitialUFace1D([](std::size_t, double) { return 0.0; });
  }

  wrap.SetPDE(std::move(pde));

  const int err = wrap.NumericalSolve();
  if (err) {
    std::cerr << "1D Solve failed with code " << err << "\n";
    return err;
  }

  const std::size_t s_eta = clamp_stride(wrap.Eta1D().size(), 14);
  const std::size_t s_u = clamp_stride(wrap.UFace1D().size(), 14);

  print_1d_sample("eta", wrap.Eta1D(), s_eta);
  print_1d_sample("u", wrap.UFace1D(), s_u);

  return 0;
}
