#pragma once

#include <array>
#include <utility>

#include "IterTools.hpp"

namespace fcfd::itertools
{

template<typename Index>
class SquareRange
{
public:
  SquareRange(Index start, Index end)
    : m_bounds(start, end)
  {
  }

  auto GetStart() const -> Index
  {
    return m_bounds.first;
  }

  auto GetEnd() const -> Index
  {
    return m_bounds.second;
  }

  template<std::size_t Dim>
  auto operator()([[maybe_unused]] const std::array<IterState<Index>, Dim>& iterState,
    [[maybe_unused]] std::size_t curLevel) const -> std::pair<Index, Index>
  {
    // Simply return the bounds regardless of level
    return m_bounds;
  }

private:
  const std::pair<Index, Index> m_bounds;
};

}  // namespace fcfd::itertools
