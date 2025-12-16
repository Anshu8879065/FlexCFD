# Create object library for SolTools
add_library(SolTools OBJECT)

# Force C++ linker
set_target_properties(SolTools PROPERTIES LINKER_LANGUAGE CXX)

# Include directories so headers can be found
target_include_directories(SolTools
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/casulliwrap
        ${CMAKE_CURRENT_SOURCE_DIR}/petscwrap
)

# Add sources and headers
target_sources(SolTools
    PUBLIC
        Soltools.hpp
        casulliwrap/Casulliwrap.hpp
        casulliwrap/CG.hpp
        petscwrap/GetIndex.hpp
        petscwrap/IterCallbacks.hpp
        petscwrap/PetscInitException.hpp
        petscwrap/PetscOptions.hpp
        petscwrap/PetscWrap.hpp
)

# Link with dependencies
target_link_libraries(SolTools
    PUBLIC
        itertools
        pdemodel
        PETSc::PETSc
)

# Set C++ standard
target_compile_features(SolTools
    PUBLIC
        cxx_std_20
)
