#pragma once

#include <concepts>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace fcfd::pdemodel
{

// PDE classes use BoundaryType::Dirichlet / BoundaryType::Neumann
enum class BoundaryType : int
{
  Dirichlet = 0,
  Neumann   = 1
};

template<std::floating_point FloatingPointType>
struct BoundaryCondition
{
  using State      = std::vector<FloatingPointType>;
  using CoordsSpan = std::span<const FloatingPointType>;

  // --- Coords-based functors (your original design) ---
  using CoordsFunctor = std::function<State(CoordsSpan)>;

  // --- State-based functor (needed by your old SVE/SWE2d/SVEcas snippets) ---
  // Example usage:
  //   auto bc = BoundaryCondition(BoundaryType::Neumann,
  //     [](State const& u_inner, FloatingPointType t){ return u_inner; });
  using StateFunctor = std::function<State(State const&, FloatingPointType)>;

  // ------------------------------------------------------------------
  // Original fields (as previously)
  // ------------------------------------------------------------------
  CoordsFunctor condBoundFuns{};
  int boundaryType = static_cast<int>(BoundaryType::Dirichlet);
  CoordsFunctor bdryFunOp{};          // coords-based (old)
  CoordsFunctor idToBdryFunRhs{};     // coords-based (old)

  // ------------------------------------------------------------------
  // Added field: state-based BC operator (used by SVE/SWE2d/SVEcas )
  // ------------------------------------------------------------------
  StateFunctor bdryFunState{};

  BoundaryCondition() = default;

  // Constructor for the NEW/used style in your PDE classes (state-based)
  BoundaryCondition(BoundaryType type, StateFunctor fun)
    : boundaryType(static_cast<int>(type))
    , bdryFunState(std::move(fun))
  {}

  // Constructor for your ORIGINAL style (coords-based)
  BoundaryCondition(BoundaryType type, CoordsFunctor fun)
    : boundaryType(static_cast<int>(type))
    , bdryFunOp(std::move(fun))
  {}

  auto Type() const noexcept -> BoundaryType
  {
    return static_cast<BoundaryType>(boundaryType);
  }

  // ------------------------------------------------------------------
  // Helper evaluators (optional ):
  // - Prefer state-based if present
  // - Fallback to coords-based if present
  // ------------------------------------------------------------------

  auto Eval(State const& u_inner, FloatingPointType t) const -> State
  {
    if (bdryFunState) return bdryFunState(u_inner, t);
    return {}; // not set
  }

  auto Eval(CoordsSpan x) const -> State
  {
    if (bdryFunOp) return bdryFunOp(x);
    if (condBoundFuns) return condBoundFuns(x);
    if (idToBdryFunRhs) return idToBdryFunRhs(x);
    return {}; // not set
  }
};

} // namespace fcfd::pdemodel
