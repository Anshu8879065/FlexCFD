set(flexcfd_FOUND YES)

include(CMakeFindDependencyMacro)
find_dependency(fmt)

if(flexcfd_FOUND)
  include("${CMAKE_CURRENT_LIST_DIR}/flexcfdTargets.cmake")
endif()
