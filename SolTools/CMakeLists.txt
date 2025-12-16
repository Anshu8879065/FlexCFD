# Define the SolTools library
add_library(SolTools
    Soltools.hpp
)

# Add subdirectories for CasulliWrap and PetscWrap
add_subdirectory(casulliwrap)
add_subdirectory(petscwrap)

# Include directories for headers
target_include_directories(SolTools
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}          # SolTools main headers
        ${CMAKE_CURRENT_SOURCE_DIR}/casulliwrap
        ${CMAKE_CURRENT_SOURCE_DIR}/petscwrap
)

# Link dependencies
target_link_libraries(SolTools
    PUBLIC
        PdeSystem         # Your PdeSystem library
        Utils_itertools   # Your itertools utility library
        PETSc::PETSc      # PETSc library
)

# Set C++ standard
target_compile_features(SolTools
    PUBLIC
        cxx_std_20
)
