add_library(flexcfd_lib OBJECT)

set_target_properties(flexcfd_lib PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(utils)
add_subdirectory(pdes)
add_subdirectory(numerics)

target_link_libraries(flexcfd_lib
    PUBLIC
        flexcfd_utils
        flexcfd_pdemodel
        flexcfd_pdenumerics
        $<TARGET_OBJECTS:flexcfd_pdenumerics>
)

target_compile_features(flexcfd_lib PUBLIC cxx_std_20)
