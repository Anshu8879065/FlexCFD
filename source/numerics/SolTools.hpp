#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

#include "pdes/PDESystem.hpp"

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
  std::semiregular PDEOptions,
  std::integral ErrorType,
  typename MethodP,
  typename OptForm>
class SolTool
{
public:
  ///
  /// \brief Default constructor
  ///
  constexpr explicit SolTool() = default;

  ///
  /// \brief Copy constructor
  ///
  constexpr SolTool(SolTool const&) = default;

  ///
  /// \brief Copy-assignment operator
  ///
  constexpr auto operator=(SolTool const&) -> SolTool& = default;

  ///
  /// \brief Move constructor
  ///
  constexpr SolTool(SolTool&&) noexcept = default;

  ///
  /// \brief Move-assignment operator
  ///
  constexpr auto operator=(SolTool&&) noexcept -> SolTool& = default;

  ///
  /// \brief Virtual destructor
  ///
  constexpr virtual ~SolTool() noexcept = default;

  ///
  /// \brief Set the solution method
  /// \returns An ErrorType indicating success or failure
  ///
  virtual auto InitSolMethod() -> ErrorType = 0;

  ///
  /// \brief Set the solution method
  /// \param[in] solMethod The solution method
  ///
  constexpr void InitSolMethod(MethodP solMethod)
  {
    m_methodProps = std::move(solMethod);
    SetInitFlag(0, FlagState::IsSet);
  }

  ///
  /// \brief Set solution options
  /// \returns An ErrorType indicating success or failure
  ///
  virtual auto SetSolOpts() -> ErrorType = 0;

  ///
  /// \brief Set solution options
  /// \param[in] solOptions The solution options to be set
  ///
  virtual void SetSolOpts(OptForm solOptions)
  {
    m_algOpts = std::move(solOptions);
    SetInitFlag(1, FlagState::IsSet);
  }

  ///
  /// \brief Set the PDE to be solved
  /// \returns An ErrorType indicating success or failure
  ///
  virtual auto SetPDE() -> ErrorType
  {
    return 0;
  }

  ///
  /// \brief Set the PDE to be solved
  /// \param[in] pde The PDE to be solved
  ///
  constexpr void SetPDE(std::unique_ptr<pdemodel::PDESystem<FloatingPointType, PDEOptions>> pde) noexcept
  {
    m_myPDE = std::move(pde);
  }

  virtual auto NumericalSolve() -> ErrorType = 0;

  virtual auto EvaluateSolution(std::function<std::vector<double>(std::vector<FloatingPointType>&)> const& interpol)
    -> ErrorType = 0;

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
  {
    assert(idx < m_initialized.size() and "Index out of bounds");
    m_initialized.at(idx) = val;
  }

  ///
  /// \brief Get the value of an initialization flag at the given index
  /// \param[in] idx The index of the flag
  /// \returns The value of the flag
  ///
  constexpr auto GetInitFlag(std::size_t idx) const -> FlagState
  {
    assert(idx < m_initialized.size() and "Index out of bounds");
    return m_initialized.at(idx);
  }

private:
  //! An array of flags storing the initialized state of the class' member variables
  std::array<FlagState, 4> m_initialized {
    FlagState::IsNotSet, FlagState::IsNotSet, FlagState::IsNotSet, FlagState::IsNotSet};
};
}  // namespace fcfd::pdenumerics
