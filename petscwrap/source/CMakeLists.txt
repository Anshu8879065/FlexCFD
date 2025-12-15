find_package(PETSc CONFIG REQUIRED)

# ---- Declare library ----

add_library(flexcfd_lib OBJECT)

set_target_properties(flexcfd_lib PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(fcfd/itertools)
add_subdirectory(fcfd/pdemodel)
add_subdirectory(fcfd/pdenumerics)

target_include_directories(flexcfd_lib
    ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_link_libraries(flexcfd_lib
    PUBLIC
        itertools
        pdemodel
        pdenumerics
        PETSc::PETSc
)

target_compile_features(flexcfd_lib PUBLIC cxx_std_20)


# ---- Declare executable ----

add_executable(flexcfd_exe main.cpp)
add_executable(flexcfd::exe ALIAS flexcfd_exe)

set_property(TARGET flexcfd_exe PROPERTY OUTPUT_NAME flexcfd)

target_compile_features(flexcfd_exe PRIVATE cxx_std_20)

target_link_libraries(flexcfd_exe PRIVATE flexcfd_lib)
