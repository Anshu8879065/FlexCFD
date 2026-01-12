find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)

if (NOT PETSC_FOUND)
    message(FATAL_ERROR "PETSc was not found. Aborting...")
endif()

add_library(flexcfd_pdenumerics OBJECT)

target_sources(flexcfd_pdenumerics
    PUBLIC
        GetIndex.hpp
        IterCallbacks.hpp
        SolTools.hpp
        petscwrap/PetscUtils.hpp
        petscwrap/PetscWrap.hpp
        casulliwrap/CasulliMethodProps.hpp
        casulliwrap/CasulliWrap.hpp
    PRIVATE
        petscwrap/PetscUtils.cpp
)

target_link_libraries(flexcfd_pdenumerics
    PUBLIC
        flexcfd_utils
        flexcfd_pdemodel
        PkgConfig::PETSC
)

target_include_directories(flexcfd_pdenumerics
    ${warning_guard}
    PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(flexcfd_pdenumerics
    PUBLIC
        cxx_std_20
)
