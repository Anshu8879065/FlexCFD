#pragma once

#include <array>
#include <utility>

template<typename Index>
struct IterState
{
  Index lo {0};
  Index hi {0};
  Index cur {0};
};

/*
template<typename Index,
    typename Callback,
    size_t Dim,
    typename Level>
bool IterLevel(Callback&& callback,
    std::array<IterState<Index>, Dim>& iterState,
    size_t minChangeLevel,
    Level&& levelBounds)
{
    static constexpr size_t curLevel = Dim - 1;
    auto curBounds = std::invoke(levelBounds, iterState, curLevel);
    const Index stBound = curBounds.first;
    const Index endBound = curBounds.second;

    if (stBound < endBound)
    {
        iterState[curLevel].lo = stBound;
        iterState[curLevel].hi = endBound;
        iterState[curLevel].cur = stBound;


    }

    return true;
}
*/

template<typename Index,
    size_t Limit,
    typename Callback,
    size_t Dim,
    typename Level,
    typename... Rest>
bool IterLevel(Callback&& callback,
    std::array<IterState<Index>, Dim>& iterState,
    size_t minChangeLevel,
    Level&& levelBounds,
    Rest&&... restBounds)
{
  static constexpr size_t curLevel = Dim - sizeof...(Rest) - 1;

  auto curBounds = std::invoke(levelBounds, iterState, curLevel);
  const Index stBound = curBounds.first;
  const Index endBound = curBounds.second;

  if (stBound < endBound) {
    iterState[curLevel].lo = stBound;
    iterState[curLevel].hi = endBound;
    iterState[curLevel].cur = stBound;

    const Index nxBound = stBound + 1;
    if constexpr (curLevel < Limit - 1) {
      if (!IterLevel<Index, Limit>(std::forward<Callback>(callback),
              iterState,
              minChangeLevel,
              std::forward<Rest>(restBounds)...))
      {
        return false;
      }

      for (Index idx = nxBound; idx < endBound; ++idx) {
        iterState[curLevel].cur = idx;
        if (!IterLevel<Index, Limit>(std::forward<Callback>(callback),
                iterState,
                curLevel,
                std::forward<Rest>(restBounds)...))
        {
          return false;
        }
      }
    }
    else {
      const std::array<IterState<Index>, Dim>& constIterState {iterState};

      if (!std::invoke(callback, constIterState, minChangeLevel)) {
        return false;
      }

      for (Index idx = nxBound; idx < endBound; ++idx) {
        iterState[curLevel].cur = idx;
        if (!std::invoke(callback, constIterState, curLevel)) {
          return false;
        }
      }
    }
  }
  return true;
}

template<typename Index, size_t Limit, typename Callback, typename Level, typename... Levels>
bool NestedIterationGen(Callback&& callback, Level&& firstLevel, Levels&&... restLevels)
{
  static constexpr size_t Dim = sizeof...(Levels) + 1;

  std::array<IterState<Index>, Dim> iterState;

  return IterLevel<Index, Limit>(std::forward<Callback>(callback),
      iterState,
      0,
      std::forward<Level>(firstLevel),
      std::forward<Levels>(restLevels)...);
}

template<typename Index, typename Callback, typename Level, typename... Levels>
bool NestedIteration(Callback&& callback, Level&& firstLevel, Levels&&... restLevels)
{
  return NestedIterationGen<Index, sizeof...(Levels) + 1, Callback, Level, Levels...>(
      std::forward<Callback>(callback),
      std::forward<Level>(firstLevel),
      std::forward<Levels>(restLevels)...);
}

template<typename Index>
class SquareRange
{
public:
  SquareRange(Index start, Index end)
      : m_bounds(start, end)
  {
  }

  Index GetStart() const
  {
    return m_bounds.first;
  }

  Index GetEnd() const
  {
    return m_bounds.second;
  }

  template<size_t Dim>
  std::pair<Index, Index> operator()(
      const std::array<IterState<Index>, Dim>& iterState, size_t curLevel) const
  {
    // Simply return the bounds regardless of level
    return m_bounds;
  }

private:
  const std::pair<Index, Index> m_bounds;
};

template<typename Index>
class TriangleRange
{
public:
  TriangleRange(Index initial)
      : m_initial(initial)
  {
  }

  template<size_t Dim>
  std::pair<Index, Index> operator()(
      const std::array<IterState<Index>, Dim>& iterState, size_t curLevel) const
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
