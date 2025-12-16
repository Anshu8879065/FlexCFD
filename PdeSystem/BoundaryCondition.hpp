#pragma once

#include <concepts>
#include <functional>
#include <span>
#include <vector>

namespace fcfd::pdemodel
{

template<std::floating_point FloatingPointType>
struct BoundaryCondition
{
  std::function<std::vector<FloatingPointType>(std::span<const FloatingPointType>)> condBoundFuns;
  int boundaryType;
  std::function<std::vector<FloatingPointType>(std::span<const FloatingPointType>)> bdryFunOp;
  std::function<std::vector<FloatingPointType>(std::span<const FloatingPointType>)>
      idToBdryFunRhs;
};

}  // namespace fcfd::pdemodel
