#pragma once

#include <array>
#include <functional>

namespace fcfd::itertools
{

template<typename Index>
struct IterState
{
  Index lo {0};
  Index hi {0};
  Index cur {0};
};

template<typename Index, std::size_t Limit, typename Callback, std::size_t Dim, typename Level, typename... Rest>
auto IterLevel(Callback&& callback,
  std::array<IterState<Index>, Dim>& iterState,
  std::size_t minChangeLevel,
  Level&& levelBounds,
  Rest&&... restBounds) -> bool
{
  static constexpr std::size_t curLevel = Dim - sizeof...(Rest) - 1;

  auto curBounds = std::invoke(std::forward<Level>(levelBounds), iterState, curLevel);
  const Index stBound = curBounds.first;
  const Index endBound = curBounds.second;

  if (stBound < endBound) {
    iterState[curLevel].lo = stBound;
    iterState[curLevel].hi = endBound;
    iterState[curLevel].cur = stBound;

    const Index nxBound = stBound + 1;

    if constexpr (curLevel < Limit - 1) {
      if (!IterLevel<Index, Limit>(
            std::forward<Callback>(callback), iterState, minChangeLevel, std::forward<Rest>(restBounds)...))
      {
        return false;
      }

      for (Index idx = nxBound; idx < endBound; ++idx) {
        iterState[curLevel].cur = idx;

        if (!IterLevel<Index, Limit>(
              std::forward<Callback>(callback), iterState, curLevel, std::forward<Rest>(restBounds)...))
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

template<typename Index, std::size_t Limit, typename Callback, typename Level, typename... Levels>
auto NestedIterationGen(Callback&& callback, Level&& firstLevel, Levels&&... restLevels) -> bool
{
  static constexpr std::size_t Dim = sizeof...(Levels) + 1;
  std::array<IterState<Index>, Dim> iterState;

  return IterLevel<Index, Limit>(std::forward<Callback>(callback),
    iterState,
    0,
    std::forward<Level>(firstLevel),
    std::forward<Levels>(restLevels)...);
}

template<typename Index, typename Callback, typename Level, typename... Levels>
auto NestedIteration(Callback&& callback, Level&& firstLevel, Levels&&... restLevels) -> bool
{
  return NestedIterationGen<Index, sizeof...(Levels) + 1, Callback, Level, Levels...>(
    std::forward<Callback>(callback), std::forward<Level>(firstLevel), std::forward<Levels>(restLevels)...);
}

}  // namespace fcfd::itertools
