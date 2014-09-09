#ifndef __UASTAR_PUZZLE
#define __UASTAR_PUZZLE

#include "problem.hpp"

class PuzzlePrivate;
class Puzzle : public Problem {
public:
    Puzzle();
    ~Puzzle();
    string problemName() const;
    void prepare();
    void cpuInitialize();
    void gpuInitialize();
    void cpuSolve();
    void gpuSolve();
    bool output();

    bool inrange(int x, int y) {
        return 0 <= x && x < n && 0 <= y && y < n;
    }
    int length() {
        return n;
    }
    int size() {
        return n*n;
    }
    void initialState(vector<uint8_t> &data) {
        for (int i = 0; i < (int)m_initialState.size(); ++i)
            dout << m_initialState[i] << " ";
        dout << endl;
        data = m_initialState;
    }
    int tileID(int x, int y) const {
        return x * n + y;
    }
    void tileXY(int id, int *x, int *y) const {
        *x = id / n;
        *y = id % n;
    }

private:
    void printSolution(const vector<int> &solution, const string &filename) const;
    void printState(const vector<uint8_t> &state, FILE *f) const;

    int n;
    vector<uint8_t> m_initialState;
    PuzzlePrivate *d;
};

#endif
