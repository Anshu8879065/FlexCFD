#ifndef FLEXCFD_HPP
#define FLEXCFD_HPP

#include <array>
#include <cassert>
#include <cmath>
#include <execution>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/petsc.h>
#include <memory.h>
#include <petsc.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscsystypes.h>

using namespace dolfinx;
using namespace std;

#include "FlexCFD_types.hpp"  // Suggest putting template and struct definitions there

#endif  // FLEXCFD_HPP
