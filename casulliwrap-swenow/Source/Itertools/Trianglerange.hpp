#pragma once

#include <utility>

#include "IterTools.hpp"

namespace fcfd::itertools
{

template<typename Index>
class TriangleRange
{
public:
  explicit TriangleRange(Index initial)
    : m_initial(initial)
  {
  }

  template<std::size_t Dim>
  auto operator()(const std::array<IterState<Index>, Dim>& iterState, std::size_t curLevel) const
    -> std::pair<Index, Index>
  {
    // "Triangle" means once the value of the previous level is reached, stop
    // Note that if curLevel == 0, this class makes no sense for a range,
    // so we return (initial,initial)
    std::pair<Index, Index> retPair {m_initial, m_initial};

    if (curLevel > 0) {
      retPair.second = iterState[curLevel - 1].cur;
    }

    return retPair;
  }

private:
  const Index m_initial;
};

}  // namespace fcfd::itertools
