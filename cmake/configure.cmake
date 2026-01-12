include(FindPkgConfig REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if ($ENV{PETSC_DIR})
    set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
    set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
endif()

pkg_get_variable(C_COMPILER PETSc ccompiler)
pkg_get_variable(CXX_COMPILER PETSc cxxcompiler)
pkg_get_variable(FORTRAN_COMPILER PETSc fcompiler)

if (C_COMPILER)
    set(CMAKE_C_COMPILER ${C_COMPILER} CACHE STRING "C compiler")
endif()

if (CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${CXX_COMPILER} CACHE STRING "C++ compiler")
endif()

if (FORTRAN_COMPILER)
    set(CMAKE_Fortran_COMPILER ${FORTRAN_COMPILER} CACHE STRING "Fortran compiler")
endif()
