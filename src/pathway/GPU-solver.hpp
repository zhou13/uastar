#ifndef __GPU_SOLVER_HPP_1LYUPTGF
#define __GPU_SOLVER_HPP_1LYUPTGF

#include "pathway/pathway.hpp"

const int OPEN_LIST_SIZE = 10000000;
const int NODE_LIST_SIZE = 150000000;
const int ANSWER_LIST_SIZE = 50000;

const int NUM_BLOCK  = 13 * 3;
const int NUM_THREAD = 192;
const int NUM_TOTAL = NUM_BLOCK * NUM_THREAD;

const int VALUE_PER_THREAD = 1;
const int NUM_VALUE = NUM_TOTAL * VALUE_PER_THREAD;

const int HEAP_CAPACITY = OPEN_LIST_SIZE / NUM_TOTAL;

class DeviceData;
class GPUPathwaySolver {
public:
    GPUPathwaySolver(Pathway *pathway);
    ~GPUPathwaySolver();
    void initialize();
    bool solve();
    void getSolution(float *optimal, vector<int> *pathList);

private:
    bool isPrime(uint32_t number);
    vector<uint32_t> genRandomPrime(uint32_t maximum, int count);
    // Problem
    Pathway *p;
    DeviceData *d;
    uint32_t m_optimalNodeAddr;
    float m_optimalDistance;
};

#endif /* end of include guard: __GPU_SOLVER_HPP_1LYUPTGF */
