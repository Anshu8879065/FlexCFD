#pragma once

#include <exception>
#include <string>

namespace fcfd::pdenumerics
{

class PetscInitException : public std::exception
{
public:
  /// \brief Create a default PetscInitException instance
  explicit PetscInitException() noexcept = default;

  /// \brief Create a PetscInitException instance with the given message
  /// \param[in] message A description of the error that occured
  explicit PetscInitException(std::string message) noexcept
    : m_msg(std::move(message))
  {
  }

  /// \brief Returns the explanatory string
  /// \returns The explanatory string
  constexpr auto what() const noexcept -> char const* override
  {
    return m_msg.c_str();
  }

private:
  std::string m_msg;
};

}  // namespace fcfd::pdenumerics
