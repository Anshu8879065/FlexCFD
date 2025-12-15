add_library(itertools OBJECT)

set_target_properties(itertools PROPERTIES LINKER_LANGUAGE CXX)

target_sources(itertools
    PUBLIC
        IterTools.hpp
        TriangleRange.hpp
        SquareRange.hpp
)

target_compile_features(itertools PUBLIC cxx_std_20)