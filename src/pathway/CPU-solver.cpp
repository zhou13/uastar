#include "pathway/CPU-solver.hpp"

const real s2 = sqrt(2);
const int dx[8]  = { 1, 1,  1, 0,  0, -1, -1, -1 };
const int dy[8]  = { 1, 0, -1, 1, -1,  1,  0, -1 };
const float cst[8] = {s2, 1, s2, 1,  1, s2,  1, s2 };

CPUPathwaySolver::CPUPathwaySolver(Pathway *pathway)
    : p(pathway)
{
    target = vec2(p->ex(), p->ey());
    targetID = p->toID(p->ex(), p->ey());
    // pass
}

CPUPathwaySolver::~CPUPathwaySolver()
{
    for (auto &pair : globalList)
        delete pair.second;
    globalList.clear();
}

void CPUPathwaySolver::initialize()
{
    for (auto &pair : globalList)
        delete pair.second;
    globalList.clear();

    openList = std::priority_queue<pair<real, node_t *>>();
    closeList.clear();

    int id = p->toID(p->sx(), p->sy());
    node_t *startNode = new node_t(id, 0, nullptr);
    openList.push(make_pair(computeFValue(startNode), startNode));
    globalList[startNode->id] = startNode;
}

bool CPUPathwaySolver::solve(real *optimal, vector<vec2> *solution)
{
    while (!openList.empty()) {
        node_t *node;
        do {
            node = openList.top().second;
            openList.pop();
        } while (closeList.count(node->id));
        closeList.insert(node->id);

        if (node->id == targetID) {
            *optimal = node->dist;
            solution->clear();
            while (node) {
                solution->push_back(p->toVec(node->id));
                node = node->prev;
            }
            std::reverse(solution->begin(), solution->end());
            return true;
        }

        int x, y;
        p->toXY(node->id, &x, &y);
        for (int i = 0; i < 8; ++i) {
            if (~p->graph()[node->id] & 1 << i)
                continue;
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (p->inrange(nx, ny)) {
                int nid = p->toID(nx, ny);
                real dist = node->dist + cst[i];
                if (globalList.count(nid) == 0) {
                    node_t *nnode = new node_t(nid, dist, node);
                    globalList[nid] = nnode;
                    openList.push(make_pair(computeFValue(nnode), nnode));
                } else {
                    node_t *onode = globalList[nid];
                    if (dist < onode->dist) {
                        onode->dist = dist;
                        openList.push(make_pair(computeFValue(onode), onode));
                    }
                }
            }
        }
    }
    return false;
}

real CPUPathwaySolver::computeFValue(node_t *node)
{
    return node->dist + target.distance(p->toVec(node->id));
}
