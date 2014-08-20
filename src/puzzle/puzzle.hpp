#ifndef __UASTAR_PUZZLE
#define __UASTAR_PUZZLE

#include "problem.hpp"

class Puzzle : public Problem {
public:
    Puzzle();
    ~Puzzle() override;
    string problemName() const override;
    void prepare() override;
    void cpuInitialize() override;
    void gpuInitialize() override;
    void cpuSolve() override;
    void gpuSolve() override;
    bool output() const override;
private:
};

#endif
