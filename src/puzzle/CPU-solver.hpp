#ifndef __CPU_SOLVER_HPP_IJ1ERDAZ
#define __CPU_SOLVER_HPP_IJ1ERDAZ

#include "puzzle/puzzle.cuh"
#include "puzzle/storage.hpp"
#include "puzzle/database.hpp"

#include <queue>
#include <boost/unordered_map.hpp>

namespace cpusolver {

template<int N>
struct node_t {
    PuzzleStorage<N> ps;
    node_t<N> *prev;
    int fValue;
    int gValue;
};

template<int N>
struct heap_t {
    int fValue;
    node_t<N> *node;
    bool operator<(const heap_t<N> &r) const {
        return fValue < r.fValue;
    }
    bool operator>(const heap_t<N> &r) const {
        return fValue > r.fValue;
    }
};

template <int N>
class CPUPuzzleSolver {
public:
    CPUPuzzleSolver(Puzzle *puzzle) : p(puzzle) { }
    ~CPUPuzzleSolver() {
        for (typename boost::unordered_map<
             PuzzleStorage<N>, node_t<N> * >::iterator
             it = closeList.begin(); it != closeList.end(); ++it)
            delete it->second;
    }
    void initialize() {
        if (N == 3) {
            dbCount = 1;
            tracked.resize(dbCount);
            database.resize(dbCount);
            for (int i = 1; i <= 8; ++i)
                tracked[0].push_back(i);
        } else if (N == 4) {
            dbCount = 2;
            tracked.resize(dbCount);
            database.resize(dbCount);

            tracked[0].push_back(1);
            tracked[0].push_back(2);
            tracked[0].push_back(3);
            tracked[0].push_back(4);
            tracked[0].push_back(5);
            tracked[0].push_back(6);
            tracked[0].push_back(9);
            tracked[0].push_back(13);

            tracked[1].push_back(7);
            tracked[1].push_back(8);
            tracked[1].push_back(10);
            tracked[1].push_back(11);
            tracked[1].push_back(12);
            tracked[1].push_back(14);
            tracked[1].push_back(15);
        } else if (N == 5) {
            dbCount = 4;
            tracked.resize(dbCount);
            database.resize(dbCount);

            tracked[0].push_back(3);
            tracked[0].push_back(4);
            tracked[0].push_back(5);
            tracked[0].push_back(10);
            tracked[0].push_back(15);
            tracked[0].push_back(20);

            tracked[1].push_back(2);
            tracked[1].push_back(1);
            tracked[1].push_back(6);
            tracked[1].push_back(11);
            tracked[1].push_back(16);
            tracked[1].push_back(21);

            tracked[2].push_back(7);
            tracked[2].push_back(8);
            tracked[2].push_back(9);
            tracked[2].push_back(12);
            tracked[2].push_back(17);
            tracked[2].push_back(22);

            tracked[3].push_back(13);
            tracked[3].push_back(14);
            tracked[3].push_back(18);
            tracked[3].push_back(19);
            tracked[3].push_back(23);
            tracked[3].push_back(24);
        } else
            assert(false);

        index.resize(dbCount);
        multiple.resize(dbCount);
        mapTracked.resize(N*N, make_pair(-1, -1));
        for (int i = 0; i < dbCount; ++i) {
            PatternDatabase pd(N, tracked[i]);
            database[i].resize(pd.size());
            pd.genDatabase(database[i].data());

            index[i].resize(tracked[i].size());
            multiple[i].reserve(tracked[i].size() + 1);

            for (int j = 0; j < (int)tracked[i].size(); ++j) {
                mapTracked[tracked[i][j]] = make_pair(i, j);
                multiple[i].push_back(N*N - j);
            }
            multiple[i].push_back(1);

            for (int j = tracked[i].size()-2; j >= 0; --j)
                multiple[i][j] *= multiple[i][j+1];
        }

        node_t<N> *node = new node_t<N>;
        vector<uint8_t> state;
        p->initialState(state);

        node->ps = PuzzleStorage<N>(
            *reinterpret_cast<uint8_t(*)[N][N]>(state.data()));
        node->prev = 0;
        node->fValue = computeHValue(
            *reinterpret_cast<uint8_t(*)[N][N]>(state.data()));
        node->gValue = 0;

        heap_t<N> heapItem;
        heapItem.fValue = node->fValue;
        heapItem.node = node;

        openList.push(heapItem);
        closeList[node->ps] = node;

        int count = 0;
        uint8_t _targetState[N][N];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                _targetState[i][j] = ++count;
        _targetState[N-1][N-1] = 0;
        targetState = PuzzleStorage<N>(_targetState);
    }

    bool solve() {
        uint8_t conf[N][N];

        int round = 0;
        while (!openList.empty()) {
            dprintf(" ======Round %d=======\n", round);
            heap_t<N> now;
            do {
                now = openList.top();
                openList.pop();
            } while (closeList[now.node->ps]->fValue != now.fValue);

            node_t<N> *node = now.node;
            if (node->ps == targetState) {
                optimalNode = node;
                return true;
            }

            if (DEBUG_CONDITION) {
                dout << "\tfValue: " << node->fValue << endl;
                dout << "\tgValue: " << node->gValue << endl;
                printStorage(node->ps, "\t");
            }

            int x, y;
            node->ps.decompose(conf);
            getEmptyTile(conf, &x, &y);

            for (int i = 0; i < 4; ++i) {
                int nx = x + DX[i];
                int ny = y + DY[i];
                if (p->inrange(nx, ny)) {
                    std::swap(conf[x][y], conf[nx][ny]);

                    node_t<N> nnode;
                    nnode.ps = PuzzleStorage<N>(conf);
                    nnode.gValue = node->gValue + 1;
                    nnode.fValue = nnode.gValue + computeHValue(conf);
                    nnode.prev = node;

                    if (DEBUG_CONDITION) {
                        dout << "\t\tfValue: " << nnode.fValue << endl;
                        dout << "\t\tgValue: " << nnode.gValue << endl;
                        printStorage(nnode.ps, "\t\t");
                        dout << endl;
                    }

                    heap_t<N> heapItem;
                    heapItem.fValue = nnode.fValue;

                    if (closeList.count(nnode.ps) == 0) {
                        node_t<N> *inode = new node_t<N>(nnode);
                        closeList[nnode.ps] = inode;
                        heapItem.node = inode;

                        openList.push(heapItem);
                    } else {
                        node_t<N> *onode = closeList[nnode.ps];
                        if (onode->fValue > nnode.fValue) {
                            *onode = nnode;
                            heapItem.node = onode;

                            openList.push(heapItem);
                        }
                    }
                    std::swap(conf[x][y], conf[nx][ny]);
                }
            }
            ++round;
        }
        return false;
    }

    void getSolution(int *optimal, vector<int> *pathList) {
        *optimal = optimalNode->fValue;

        node_t<N> *curr = optimalNode;
        node_t<N> *prev = curr->prev;

        uint8_t cconf[N][N];
        uint8_t pconf[N][N];

        pathList->clear();
        while (prev) {
            curr->ps.decompose(cconf);
            prev->ps.decompose(pconf);

            int cx, cy;
            int px, py;
            getEmptyTile(cconf, &cx, &cy);
            getEmptyTile(pconf, &px, &py);
            for (int i = 0; i < 4; ++i)
                if (px + DX[i] == cx && py + DY[i] == cy)
                    pathList->push_back(i);

            curr = prev;
            prev = curr->prev;
        }

        std::reverse(pathList->begin(), pathList->end());
    }

private:
    typedef typename std::priority_queue<
        heap_t<N>,
        vector< heap_t<N> >,
        std::greater< heap_t<N> > > openlist_t;

    typedef typename boost::unordered_map<
        PuzzleStorage<N>,
        node_t<N> * > closelist_t;

    Puzzle *p;

    openlist_t openList;
    closelist_t closeList;

    PuzzleStorage<N> targetState;
    node_t<N> *optimalNode;

    int dbCount;
    vector< vector<uint8_t> > database;
    vector< vector<int> > tracked;
    vector< pair<int, int> > mapTracked;
    vector< vector<int> > index;
    vector< vector<int> > multiple;

    int computeHValue(uint8_t conf[N][N]) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (conf[i][j]) {
                    pair<int, int> p = mapTracked[conf[i][j]];
                    index[p.first][p.second] = i*N + j;
                }

        int retn = 0;
        for (int k = 0; k < dbCount; ++k) {
            int oindex[20];
            std::copy(index[k].begin(), index[k].end(), oindex);

            for (int i = 0; i < (int)index[k].size(); ++i)
                for (int j = i+1; j < (int)index[k].size(); ++j)
                    if (oindex[i] < oindex[j])
                        --index[k][j];
            int code = 0;
            for (int i = 0; i < (int)index[k].size(); ++i)
                 code += index[k][i] * multiple[k][i+1];

            retn += database[k][code];
        }

        return retn;
    }

    void getEmptyTile(uint8_t conf[N][N], int *x, int *y) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (conf[i][j] == 0) {
                    *x = i;
                    *y = j;
                    return;
                }
    }
};

}

#endif /* end of include guard: __CPU_SOLVER_HPP_IJ1ERDAZ */
