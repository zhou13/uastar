#ifndef __CPU_SOLVER_HPP_FXKHT6GB
#define __CPU_SOLVER_HPP_FXKHT6GB

#include "pathway/pathway.hpp"
#include "vec2.hpp"
#include <queue>
#include <unordered_set>
#include <unordered_map>

class node_t;
class CPUPathwaySolver {
public:
    CPUPathwaySolver(Pathway *pathway);
    ~CPUPathwaySolver();
    void initialize();
    bool solve(float *optimal, vector<vec2> *solution);

private:
    float computeFValue(node_t *node);

    Pathway *p;
    std::priority_queue<
        pair<float, node_t *>,
        vector<pair<float, node_t *>>,
        std::greater<pair<float, node_t *>>> openList;
    std::unordered_set<int> closeList;
    std::unordered_map<int, node_t *> globalList;

    vec2 target;
    int targetID;
};

#endif /* end of include guard: __CPU_SOLVER_HPP_FXKHT6GB */

