#pragma once

#include <array>
#include <cstddef>
#include "PDEType.hpp"

namespace fcfd::pdemodel
{

template<std::size_t Dim>
struct GridParams
{
    PDEType type;

    // Number of primary cells (Nx, Ny, Nz)
    std::array<std::size_t, Dim> numCells;

    // Physical side lengths (Lx, Ly, Lz)
    std::array<double, Dim> sideLengths;

    // dx, dy, dz (computed automatically)
    std::array<double, Dim> spacings;

    double totalT;
    double dt;

    void compute_spacings() {
        for(std::size_t d=0; d<Dim; ++d)
            spacings[d] = sideLengths[d] / numCells[d];
    }
};

using Grid1dParams = GridParams<1>;
using Grid2dParams = GridParams<2>;
using Grid3dParams = GridParams<3>;

} // namespace fcfd::pdemodel

