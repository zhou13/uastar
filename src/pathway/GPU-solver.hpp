#ifndef __GPU_SOLVER_HPP_1LYUPTGF
#define __GPU_SOLVER_HPP_1LYUPTGF

#include "pathway/pathway.hpp"
#include "vec2.hpp"

class GPUPathwaySolver {
public:
    GPUPathwaySolver(Pathway *pathway);
    void initialize();
    void solve(real *optimal, vector<vec2> *solution);

private:
    // Problem
    Pathway *p;
};

#endif /* end of include guard: __GPU_SOLVER_HPP_1LYUPTGF */
