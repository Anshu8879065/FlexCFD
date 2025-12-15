install(
    TARGETS flexcfd_exe
    RUNTIME COMPONENT flexcfd_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
