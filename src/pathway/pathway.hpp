#ifndef __UASTAR_PATHWAY
#define __UASTAR_PATHWAY

#include "vec2.hpp"
#include "problem.hpp"
#include "pathway/input.hpp"

class CPUPathwaySolver;
class GPUPathwaySolver;

class Pathway : public Problem {
public:
    Pathway();
    ~Pathway();
    string problemName() const;
    void prepare();
    void cpuInitialize();
    void gpuInitialize();
    void cpuSolve();
    void gpuSolve();
    bool output() const;

    int sx() const;
    int sy() const;
    int ex() const;
    int ey() const;
    int size() const;
    int width() const;
    int height() const;
    int toID(int x, int y) const;
    void toXY(int id, int *x, int *y) const;
    vec2 toVec(int id) const;
    bool inrange(int x, int y) const;
    const uint8_t *graph() const;

private:
    void generateGraph(PathwayInput &input);

    int m_sx;
    int m_sy;
    int m_ex;
    int m_ey;
    int m_size;
    int m_width;
    int m_height;
    string m_inputModule;
    vector<uint8_t> m_graph;
    CPUPathwaySolver *cpuSolver;
    GPUPathwaySolver *gpuSolver;

    bool cpuSolved;
    bool cpuSuccessful;
    real cpuOptimal;
    vector<vec2> cpuSolution;

    bool gpuSolved;
    bool gpuSuccessful;
    real gpuOptimal;
    vector<vec2> gpuSolution;
};

inline int Pathway::sx() const
{
    return m_sx;
}

inline int Pathway::sy() const
{
    return m_sy;
}

inline int Pathway::ex() const
{
    return m_ex;
}

inline int Pathway::ey() const
{
    return m_ey;
}

inline int Pathway::size() const
{
    return m_size;
}

inline int Pathway::width() const
{
    return m_width;
}

inline int Pathway::height() const
{
    return m_height;
}

inline int Pathway::toID(int x, int y) const
{
    return x * width() + y;
}

inline void Pathway::toXY(int id, int *x, int *y) const
{
    *x = id / width();
    *y = id % width();
}

inline vec2 Pathway::toVec(int id) const
{
    return vec2(id / width(), id % width());
}

inline bool Pathway::inrange(int x, int y) const
{
    return 0 <= x && x < height() && 0 <= y && y < width();
}
    

inline const uint8_t *Pathway::graph() const
{
    return m_graph.data();
}

#endif
