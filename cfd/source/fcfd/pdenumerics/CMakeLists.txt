add_library(pdenumerics OBJECT)

set_target_properties(pdenumerics PROPERTIES LINKER_LANGUAGE CXX)

target_sources(pdenumerics
    PUBLIC
        GetIndex.hpp
        IterCallbacks.hpp
        PetscWrap.hpp
        SolTools.hpp
)

target_link_libraries(pdenumerics
    PUBLIC
        itertools
        pdemodel
        PETSc::PETSc
)

target_compile_features(pdenumerics
    PUBLIC
        cxx_std_20
)
