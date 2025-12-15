# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(flexcfdTests LANGUAGES CXX)

find_package(Catch2 REQUIRED)

include(Catch)

# ---- Tests ----

add_executable(flexcfd_test)

target_sources(flexcfd_test
    PRIVATE
        source/flexcfd_test.cpp
)

target_link_libraries(flexcfd_test
    PRIVATE
        Catch2::Catch2WithMain
        flexcfd_lib
)

target_compile_features(flexcfd_test PRIVATE cxx_std_20)

catch_discover_tests(flexcfd_test)

# ---- End-of-file commands ----

add_folders(Test)
