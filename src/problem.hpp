#ifndef __UASTAR_PROBLEM
#define __UASTAR_PROBLEM

#include "utils.hpp"

class Problem {
public:
    virtual ~Problem() { }
    // Return the problem's name
    virtual string problemName() const = 0;
    // Generate/prepare the problem's input data
    virtual void prepare() = 0;
    // Initialize the environemnt for CPU execution
    virtual void cpuInitialize() = 0;
    // Initialize the environemnt for GPU execution
    virtual void gpuInitialize() = 0;
    // Solve the problem on CPU.  Return the used wall time .
    virtual void cpuSolve() = 0;
    // Solve the problem on GPU.  Return the used wall time .
    virtual void gpuSolve() = 0;
    // Print the output.  Return whether the CPU's solution and the GPU's
    // solution is consistent.
    virtual bool output() const = 0;
};

#endif
