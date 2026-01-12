#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include "GetIndex.hpp"
#include "utils/IterTools.hpp"

namespace fcfd::pdenumerics
{

template<std::size_t Dims, std::floating_point NativeReal, std::integral NativeInt>
class BaseIterCallback
{
public:
  void SetWrapper(std::function<void(NativeReal*, NativeReal*)>* pwp)
  {
    m_funcPass = pwp;
  }

  void SetNF(NativeInt nf)
  {
    m_nf = nf;
  }

  void SetAY(NativeReal* aY)
  {
    m_aY = aY;
  }

  void SetAG(NativeReal* aG)
  {
    m_aG = aG;
  }

protected:
  std::function<void(NativeReal*, NativeReal*)> m_funcPass;
  NativeInt m_nf {};
  NativeReal* m_aY = nullptr;
  NativeReal* m_aG = nullptr;
};

template<std::size_t Dims, std::floating_point NativeReal, std::integral NativeInt>
class IterCallback : public BaseIterCallback<Dims, NativeReal, NativeInt>
{
  using BaseClass = BaseIterCallback<Dims, NativeReal, NativeInt>;
  using BaseClass::m_aG;
  using BaseClass::m_aY;
  using BaseClass::m_funcPass;
  using BaseClass::m_nf;

public:
  auto operator()(std::array<itertools::IterState<NativeInt>, Dims> const& iterState, std::size_t minChangeLevel)
    -> bool
  {
    auto const idxStart = GetIndex<Dims>::StartIndex(iterState, m_nf);
    auto const idxEnd = GetIndex<Dims>::EndIndex(iterState, m_nf);

    m_inTmp.assign(&m_aY[idxStart], &m_aY[idxEnd]);
    m_outTmp.clear();
    m_funcPass(m_inTmp, m_outTmp);

    std::ranges::copy(m_outTmp.begin(), m_outTmp.end(), &m_aG[idxStart]);
    return true;
  }

private:
  std::vector<NativeReal> m_inTmp;
  std::vector<NativeReal> m_outTmp;
};

/// This very weird looking type-traits struct actually just says, give me a function to bind
/// a specific dimensions value while forwarding the given args. It's quite reminiscent of lexical
/// scoping in functional languages

template<template<std::size_t, typename...> typename IterCallback, typename... Args>
struct CallbackTypeFunction
{
  template<std::size_t Dims>
  struct GetType
  {
    using Type = IterCallback<Dims, Args...>;
  };
};

}  // namespace fcfd::pdenumerics
