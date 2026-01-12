# Declare an executable target for the PETSc-SVE example

add_executable(petsc_sve)

target_sources(petsc_sve
  PRIVATE
    petscwrap/SVE.cpp
)

target_compile_features(petsc_sve
  PRIVATE
    cxx_std_20
)

target_link_libraries(petsc_sve
  PRIVATE
    flexcfd_lib
)

# Declare an executable target for the PETSc-SWE2d example

add_executable(petsc_swe2d)

target_sources(petsc_swe2d
  PRIVATE
    petscwrap/SWE2d.cpp
)

target_compile_features(petsc_swe2d
  PRIVATE
    cxx_std_20
)

target_link_libraries(petsc_swe2d
  PRIVATE
    flexcfd_lib
)

# Declare an executable target for the Casulli examples

add_executable(casulli1d)

target_sources(casulli1d
  PRIVATE
    Casulliwrap/SVECas.cpp
)

target_compile_definitions(casulli1d
  PRIVATE
    CASE_1D
)

target_compile_features(casulli1d
  PRIVATE
    cxx_std_20
)

target_link_libraries(casulli1d
  PRIVATE
    flexcfd_lib
)

add_executable(casulli2d)

target_sources(casulli2d
  PRIVATE
    Casulliwrap/SWE2dCas.cpp
)

target_compile_definitions(casulli2d
  PRIVATE
    CASE_2D
)

target_compile_features(casulli2d
  PRIVATE
    cxx_std_20
)

target_link_libraries(casulli2d
  PRIVATE
    flexcfd_lib
)
