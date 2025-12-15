#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

#include "fcfd/pdemodel/PDESystem.hpp"

namespace fcfd::pdenumerics
{

///
/// \enum FlagState encapsulates whether a flag has been set or not
///
enum class FlagState : uint8_t
{
  IsSet,
  IsNotSet
};

template<std::floating_point FloatingPointType,
  typename PDEOptions,
  typename ErrorType,
  typename MethodP,
  typename OptForm>
class SolTool
{
public:
  ///
  /// \brief Default constructor
  ///
  explicit SolTool() = default;

  ///
  /// \brief Set the solution method
  /// \returns An ErrorType indicating success or failure
  ///
  virtual auto InitSolMethod() -> ErrorType;

  ///
  /// \brief Set the solution method
  /// \param[in] solMethod The solution method
  ///
  constexpr void InitSolMethod(std::unique_ptr<MethodP> solMethod)
  {
    m_methodProps = std::move(solMethod);
    SetInitFlag(0, FlagState::IsSet);
  }

  virtual auto InitProblem() -> ErrorType;

  ///
  /// \brief Set solution options
  /// \returns An ErrorType indicating success or failure
  ///
  virtual auto SetSolOpts() -> ErrorType;

  ///
  /// \brief Set solution options
  /// \param[in] solOptions The solution options to be set
  ///
  constexpr void SetSolOpts(std::unique_ptr<OptForm> solOptions) noexcept
  {
    m_algOpts = std::move(solOptions);
    SetInitFlag(1, FlagState::IsSet);
  }

  ///
  /// \brief Set the PDE to be solved
  /// \returns An ErrorType indicating success or failure
  ///
  virtual auto SetPDE() -> ErrorType;

  ///
  /// \brief Set the PDE to be solved
  /// \param[in] pde The PDE to be solved
  ///
  constexpr void SetPDE(
    std::unique_ptr<pdemodel::PDESystem<FloatingPointType, PDEOptions>> pde) noexcept
  {
    m_myPDE = std::move(pde);
  }

  virtual auto NumericalSolve() -> ErrorType;

  virtual auto InterpolateSol(
    std::function<std::vector<double>(std::vector<FloatingPointType>)>& interpol) -> ErrorType;

protected:
  MethodP m_methodProps;
  OptForm m_algOpts;
  int m_nd {};

  //! A pointer to the PDE we're trying to solve
  std::unique_ptr<pdemodel::PDESystem<FloatingPointType, PDEOptions>> m_myPDE;

  ///
  /// \brief Set an initialization flag at the given index to the given value
  /// \param[in] idx The index of the flag to be set in the boolean array
  /// \param[in] val The value to set the flag to
  ///
  constexpr void SetInitFlag(std::size_t idx, FlagState val)
    noexcept(noexcept(idx >= 0 and idx < m_initialized.size()))
  {
    assert(idx >= 0 and idx < m_initialized.size() and "Index out of bounds");
    m_initialized.at(idx) = val;
  }

  ///
  /// \brief Get the value of an initialization flag at the given index
  /// \param[in] idx The index of the flag
  /// \returns The value of the flag
  ///
  constexpr auto GetInitFlag(std::size_t idx) const
    noexcept(noexcept(idx >= 0 and idx < m_initialized.size())) -> FlagState
  {
    assert(idx >= 0 and idx < m_initialized.size() and "Index out of bounds");
    return m_initialized.at(idx);
  }

private:
  //! An array of flags storing the initialized state of the class' member variables
  std::array<FlagState, 4> m_initialized {
    FlagState::IsNotSet, FlagState::IsNotSet, FlagState::IsNotSet, FlagState::IsNotSet};
};
}  // namespace fcfd::pdenumerics
