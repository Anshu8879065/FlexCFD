#pragma once

#include <array>
#include <concepts>
#include <cstddef>

#include "utils/IterTools.hpp"

namespace fcfd::pdenumerics
{

template<std::size_t Dims>
struct GetIndex;

// NOLINTBEGIN(readability-identifier-length)

template<>
struct GetIndex<1>
{
  template<std::floating_point NativeReal, std::integral NativeInt>
  static auto StartIndex(const std::array<itertools::IterState<NativeInt>, 1>& iterState, int nf) -> std::size_t
  {
    const NativeInt i = iterState[0].cur;
    return i * nf;
  }

  template<std::floating_point NativeReal, std::integral NativeInt>
  static auto EndIndex(const std::array<itertools::IterState<NativeInt>, 1>& iterState, int nf) -> std::size_t
  {
    const NativeInt i = iterState[0].cur;
    return (i + 1) * nf;
  }
};

template<>
struct GetIndex<2>
{
  template<std::floating_point NativeReal, std::integral NativeInt>
  static auto StartIndex(const std::array<itertools::IterState<NativeInt>, 2>& iterState, int nf) -> std::size_t
  {
    const NativeInt i = iterState[0].cur;
    const NativeInt ni = iterState[0].hi - iterState[0].lo;
    const NativeInt j = iterState[1].cur;

    return (i + ni * j) * nf;
  }

  template<std::floating_point NativeReal, std::integral NativeInt>
  static auto EndIndex(const std::array<itertools::IterState<NativeInt>, 2>& iterState, int nf) -> std::size_t
  {
    const NativeInt i = iterState[0].cur;
    const NativeInt ni = iterState[0].hi - iterState[0].lo;
    const NativeInt j = iterState[1].cur;

    return (i + 1 + ni * j) * nf;
  }
};

template<>
struct GetIndex<3>
{
  template<std::floating_point NativeReal, std::integral NativeInt>
  static auto StartIndex(const std::array<itertools::IterState<NativeInt>, 3>& iterState, int nf) -> std::size_t
  {
    const NativeInt i = iterState[0].cur;
    const NativeInt ni = iterState[0].hi - iterState[0].lo;
    const NativeInt j = iterState[1].cur;
    const NativeInt nj = iterState[1].hi - iterState[1].lo;
    const NativeInt k = iterState[2].cur;

    return (i + ni * (j + nj * k)) * nf;
  }

  template<std::floating_point NativeReal, std::integral NativeInt>
  static auto EndIndex(const std::array<itertools::IterState<NativeInt>, 3>& iterState, int nf) -> std::size_t
  {
    const NativeInt i = iterState[0].cur;
    const NativeInt ni = iterState[0].hi - iterState[0].lo;
    const NativeInt j = iterState[1].cur;
    const NativeInt nj = iterState[1].hi - iterState[1].lo;
    const NativeInt k = iterState[2].cur;

    return (i + 1 + ni * (j + nj * k)) * nf;
  }
};

// NOLINTEND(readability-identifier-length)

}  // namespace fcfd::pdenumerics
