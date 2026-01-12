cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)
include(cmake/configure.cmake)

project(
    flexcfd
    VERSION 0.1.0
    DESCRIPTION "Short description"
    HOMEPAGE_URL "https://example.com/"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# -------- Libraries and executables -------

add_subdirectory(source)
add_subdirectory(examples)

# ---- Install rules ----

# if(NOT CMAKE_SKIP_INSTALL_RULES)
#   include(cmake/install-rules.cmake)
# endif()

# ---- Developer mode ----

if(NOT flexcfd_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of flexcfd"
  )
endif()

include(cmake/dev-mode.cmake)
