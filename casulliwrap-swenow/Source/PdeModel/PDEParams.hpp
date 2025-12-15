#pragma once

namespace fcfd::pdemodel
{

template<typename PDEOptions>
struct PDEParams
{
  PDEOptions myopts;
  int nd {};
  int nv {};
  int nbound {};

  explicit PDEParams(int nd, int nv) noexcept
    : nd(nd)
    , nv(nv)
  {
  }
};

}  // namespace fcfd::pdemodel
