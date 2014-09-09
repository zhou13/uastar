#ifndef __GPU_SOLVER_HPP_SMX0IKQ3
#define __GPU_SOLVER_HPP_SMX0IKQ3

#include <moderngpu.cuh>

#include "puzzle/puzzle.cuh"
#include "puzzle/GPU-kernel.cuh"

namespace gpusolver {

const int OPEN_LIST_SIZE = 1000000;
const int NODE_LIST_SIZE = 15000000;
const int ANSWER_LIST_SIZE = 50000;
const int HASH_NUMBER = 90000089;

const int NUM_BLOCK  = 13 * 3;
const int NUM_THREAD = 192;
const int NUM_TOTAL = NUM_BLOCK * NUM_THREAD;

const int HEAP_CAPACITY = OPEN_LIST_SIZE / NUM_TOTAL;

int div_up(int x, int y) { return (x-1) / y + 1; }

template<int N>
struct DeviceData {
    // pattern database
    MGPU_MEM(uint8_t) database;

    // store open list + close list
    MGPU_MEM(node_t<N>) nodes;
    MGPU_MEM(int) nodeSize;

    // hash table for `nodes'
    MGPU_MEM(uint32_t) hash;

    // store open list
    MGPU_MEM(heap_t) openList;
    // store the size for each heap
    MGPU_MEM(int) heapSize;
    MGPU_MEM(int) heapBeginIndex;

    MGPU_MEM(heap_t) heapInsertList;
    MGPU_MEM(int) heapInsertSize;

    MGPU_MEM(uint32_t) optimalStep;
    MGPU_MEM(heap_t) optimalNodes;
    MGPU_MEM(int) optimalNodesSize;

    MGPU_MEM(int) answerList;
    MGPU_MEM(int) answerSize;

    typename mgpu::ContextPtr context;
};

template<int N>
class GPUPuzzleSolver {
public:
    GPUPuzzleSolver(Puzzle *puzzle) : p(puzzle) {
        d = new DeviceData<N>();
    }
    ~GPUPuzzleSolver() {
        delete d;
    }
    void initialize() {
        int dbCount;
        vector< vector<int> > tracked;

        if (N == 3) {
            dbCount = 1;
            tracked.resize(dbCount);
            for (int i = 1; i <= 8; ++i)
                tracked[0].push_back(i);
        } else if (N == 4) {
            dbCount = 2;
            tracked.resize(dbCount);

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

        vector<int2> mapTracked;
        vector<uint8_t> database;
        mapTracked.resize(N*N, make_int2(-1, -1));
        for (int i = 0; i < dbCount; ++i) {
            PatternDatabase pd(N, tracked[i]);
            int offset = database.size();
            database.resize(offset + pd.size());
            pd.fetchDatabase(database.data() + offset);
            for (int j = 0; j < (int)tracked[i].size(); ++j) {
                mapTracked[tracked[i][j]] = make_int2(i, j);
            }
        }

        cudaDeviceSynchronize();
        cudaDeviceReset();
        d->context = CreateCudaDevice(vm_options["ordinal"].as<int>());

        uint8_t s3[3][3], c3 = 0;
        uint8_t s4[4][4], c4 = 0;
        uint8_t s5[5][5], c5 = 0;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                if (i < 3 && j < 3)
                    s3[i][j] = ++c3;
                if (i < 4 && j < 4)
                    s4[i][j] = ++c4;
                s5[i][j] = ++c5;
            }
        PuzzleStorage<3> ps3(s3);
        PuzzleStorage<4> ps4(s4);
        PuzzleStorage<5> ps5(s5);
        initializeCUDAConstantMemory<N>(mapTracked.data(), ps3, ps4, ps5);

        d->database = d->context->template Malloc<uint8_t>(database);

        d->nodes = d->context->template Malloc< node_t<N> >(NODE_LIST_SIZE);
        d->nodeSize = d->context->template Fill<int>(1, 1);

        d->hash = d->context->template Fill<uint32_t>(HASH_NUMBER, UINT32_MAX);

        d->openList = d->context->template Malloc<heap_t>(OPEN_LIST_SIZE);
        d->heapSize = d->context->template Fill<int>(NUM_TOTAL, 0);
        d->heapBeginIndex = d->context->template Fill<int>(1, 0);

        d->heapInsertList = d->context->template Malloc<heap_t>(NUM_TOTAL * 4);
        d->heapInsertSize = d->context->template Fill<int>(1, 0);

        d->optimalStep = d->context->template Fill<uint32_t>(1, UINT32_MAX);
        d->optimalNodes = d->context->template Malloc<heap_t>(NUM_TOTAL);
        d->optimalNodesSize = d->context->template Fill<int>(1, 0);

        d->answerList = d->context->template Malloc<int>(ANSWER_LIST_SIZE);
        d->answerSize = d->context->template Fill<int>(1, 0);

        vector<uint8_t> state;
        p->initialState(state);
        kInitialize<N, HASH_NUMBER> <<<1, 1>>>(
            PuzzleStorage<N>(*reinterpret_cast<uint8_t(*)[N][N]>(state.data())),
            *d->database,
            *d->nodes,
            *d->hash,
            *d->openList,
            *d->heapSize
        );
#ifdef KERNEL_LOG
            cudaDeviceSynchronize();
#endif
        dout << "\t\tGPU Initialization finishes" << endl;
    }

    bool solve() {
        std::priority_queue< heap_t, vector<heap_t>, std::greater<heap_t> > pq;
        for (int round = 0; ;++round) {
            dprintf("\t\tRound %d: kExtractExpand\n", round);
            kExtractExpand<
                N, NUM_BLOCK, NUM_THREAD, HEAP_CAPACITY, HASH_NUMBER> <<<
                NUM_BLOCK, NUM_THREAD>>>(
                    *d->database,

                    *d->nodes,
                    *d->nodeSize,

                    *d->openList,
                    *d->heapSize,

                    *d->hash,

                    *d->optimalStep,
                    *d->optimalNodes,
                    *d->optimalNodesSize,

                    *d->heapInsertList,
                    *d->heapInsertSize,

                    // reset them BTW
                    *d->heapBeginIndex
                );
#ifdef KERNEL_LOG
            cudaDeviceSynchronize();
#endif
            dprintf("\t\tRound %d: Fetch optimalNodesSize: ", round);
            int optimalNodesSize = d->optimalNodesSize->Value();
            dprintf("%d\n", optimalNodesSize);

            if (optimalNodesSize) {
                printf("\t\tRound %d: Found one solution\n", round);
                vector<heap_t> optimalNodes;
                d->optimalNodes->ToHost(optimalNodes, optimalNodesSize);

                for (size_t i = 0; i != optimalNodes.size(); ++i) {
                    dprintf("\t\t\t optimalNodes's fValue: %d\n", (int)optimalNodes[i].fValue);
                    pq.push(optimalNodes[i]);
                }

            }

            uint32_t optimalStep = d->optimalStep->Value();
            dprintf("\t\t\t Global optimal step: %d\n", optimalStep);
            if (!pq.empty() && pq.top().fValue <= optimalStep) {
                printf("\t\t\t Number of nodes expanded: %d\n", d->nodeSize->Value());
                m_optimalNodeAddr = pq.top().addr;
                m_optimalStep = pq.top().fValue;
                dprintf("\t\t\t Optimal nodes address: %d\n", m_optimalNodeAddr);
                return true;
            }

            dprintf("\t\tRound %d: kHeapInsert\n", round);
            kHeapInsert<
                N, NUM_BLOCK, NUM_THREAD, HEAP_CAPACITY> <<<
                NUM_BLOCK, NUM_THREAD>>> (
                    *d->openList,
                    *d->heapSize,
                    *d->heapBeginIndex,

                    *d->heapInsertList,
                    *d->heapInsertSize,

                    // reset them BTW
                    *d->optimalStep,
                    *d->optimalNodesSize
                );
#ifdef KERNEL_LOG
            cudaDeviceSynchronize();
#endif
            int value = 0;
            d->heapInsertSize->FromHost(&value, 1);
        }
    }

    void getSolution(int *optimal, vector<int> *pathList) {
        *optimal = m_optimalStep;
        kFetchAnswer<N><<<1, 1>>>(
            *d->nodes,
            m_optimalNodeAddr,
            *d->answerList,
            *d->answerSize
        );
        int answerSize = d->answerSize->Value();
        d->answerList->ToHost(*pathList, answerSize);
        std::reverse(pathList->begin(), pathList->end());
    }

private:
    Puzzle *p;
    DeviceData<N> *d;
    uint32_t m_optimalNodeAddr;
    uint32_t m_optimalStep;
};

}

#endif /* end of include guard: __GPU_SOLVER_HPP_SMX0IKQ3 */
