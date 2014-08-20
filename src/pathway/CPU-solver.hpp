#ifndef __CPU_SOLVER_HPP_FXKHT6GB
#define __CPU_SOLVER_HPP_FXKHT6GB

#include "pathway/pathway.hpp"
#include "vec2.hpp"
#include <queue>
#include <unordered_set>
#include <unordered_map>

struct node_t {
    int id;
    real dist;
    node_t *prev;
    node_t() = default;
    node_t(int id, real dist, node_t *prev)
        : id(id), dist(dist), prev(prev) { }
};

class CPUPathwaySolver {
public:
    CPUPathwaySolver(Pathway *pathway);
    ~CPUPathwaySolver();
    void initialize();
    bool solve(real *optimal, vector<vec2> *solution);

private:
    real computeFValue(node_t *node);

    Pathway *p;
    std::priority_queue<pair<real, node_t *>> openList;
    std::unordered_set<int> closeList;
    std::unordered_map<int, node_t *> globalList;

    vec2 target;
    int targetID;
};

#endif /* end of include guard: __CPU_SOLVER_HPP_FXKHT6GB */

