add_library(pdemodel OBJECT)

set_target_properties(pdemodel PROPERTIES LINKER_LANGUAGE CXX)

target_sources(pdemodel
    PUBLIC
        BoundaryCondition.hpp
        GridParams.hpp
        ModelParams.hpp
        PDEParams.hpp
        PDESystem.hpp
        PDEType.hpp
        SVE.hpp
)

target_compile_features(pdemodel
    PUBLIC
        cxx_std_20
)