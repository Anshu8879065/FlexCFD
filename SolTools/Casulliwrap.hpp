#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <cassert>
#include "GridParams.hpp"  // your grid parameters

namespace fcfd::casulli
{

struct CasulliWrap
{
    int dim;                 // 1, 2, or 3
    int nv;                  // number of variables per cell

    std::array<std::size_t, 3> n;      // number of cells in each direction
    std::array<double, 3> dx;          // cell spacing
    std::array<double, 3> sideLengths; // domain lengths

    // cell-centered coordinates
    std::vector<double> xc, yc, zc;

    // cell-edge coordinates (staggered)
    std::vector<double> xe, ye, ze;

    // total number of DOFs
    std::size_t ndofs;

    // --- 1D SetGrid ---
    void SetGrid(const fcfd::pdemodel::Grid1dParams& gp)
    {
        dim = 1;
        n[0] = gp.numCells[0];
        sideLengths[0] = gp.sideLengths[0];
        dx[0] = gp.spacings[0];

        xc.resize(n[0]);
        xe.resize(n[0]+1);

        // compute cell centers
        for (std::size_t i=0; i<n[0]; ++i)
            xc[i] = (i + 0.5) * dx[0];

        // compute edges (i ± 1/2)
        for (std::size_t i=0; i<=n[0]; ++i)
            xe[i] = i * dx[0];

        ndofs = n[0] * nv;
    }

    // --- 2D SetGrid ---
    void SetGrid(const fcfd::pdemodel::Grid2dParams& gp)
    {
        dim = 2;
        n[0] = gp.numCells[0];
        n[1] = gp.numCells[1];

        sideLengths[0] = gp.sideLengths[0];
        sideLengths[1] = gp.sideLengths[1];

        dx[0] = gp.spacings[0];
        dx[1] = gp.spacings[1];

        xc.resize(n[0]*n[1]);
        yc.resize(n[0]*n[1]);
        xe.resize(n[0]+1);
        ye.resize(n[1]+1);

        // cell centers
        for (std::size_t j=0; j<n[1]; ++j)
            for (std::size_t i=0; i<n[0]; ++i)
            {
                xc[i + n[0]*j] = (i + 0.5) * dx[0];
                yc[i + n[0]*j] = (j + 0.5) * dx[1];
            }

        // cell edges
        for (std::size_t i=0; i<=n[0]; ++i) xe[i] = i*dx[0];
        for (std::size_t j=0; j<=n[1]; ++j) ye[j] = j*dx[1];

        ndofs = n[0]*n[1]*nv;
    }

    // --- 3D SetGrid ---
    void SetGrid(const fcfd::pdemodel::Grid3dParams& gp)
    {
        dim = 3;
        n[0] = gp.numCells[0];
        n[1] = gp.numCells[1];
        n[2] = gp.numCells[2];

        sideLengths[0] = gp.sideLengths[0];
        sideLengths[1] = gp.sideLengths[1];
        sideLengths[2] = gp.sideLengths[2];

        dx[0] = gp.spacings[0];
        dx[1] = gp.spacings[1];
        dx[2] = gp.spacings[2];

        xc.resize(n[0]*n[1]*n[2]);
        yc.resize(n[0]*n[1]*n[2]);
        zc.resize(n[0]*n[1]*n[2]);

        xe.resize(n[0]+1);
        ye.resize(n[1]+1);
        ze.resize(n[2]+1);

        // cell centers
        for (std::size_t k=0; k<n[2]; ++k)
            for (std::size_t j=0; j<n[1]; ++j)
                for (std::size_t i=0; i<n[0]; ++i)
                {
                    std::size_t idx = i + n[0]*(j + n[1]*k);
                    xc[idx] = (i + 0.5)*dx[0];
                    yc[idx] = (j + 0.5)*dx[1];
                    zc[idx] = (k + 0.5)*dx[2];
                }

        // cell edges
        for (std::size_t i=0; i<=n[0]; ++i) xe[i] = i*dx[0];
        for (std::size_t j=0; j<=n[1]; ++j) ye[j] = j*dx[1];
        for (std::size_t k=0; k<=n[2]; ++k) ze[k] = k*dx[2];

        ndofs = n[0]*n[1]*n[2]*nv;
    }

};
} // namespace fcfd::casulli

