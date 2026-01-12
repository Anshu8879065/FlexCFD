add_library(flexcfd_utils INTERFACE)

target_sources(flexcfd_utils
    INTERFACE
        IterTools.hpp
        TriangleRange.hpp
        SquareRange.hpp
)

target_include_directories(flexcfd_utils
    ${warning_guard}
    INTERFACE
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(flexcfd_utils
    INTERFACE
        cxx_std_20
)
