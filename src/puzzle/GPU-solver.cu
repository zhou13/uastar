#define NO_CPP11

#include <iostream>
#include <moderngpu.cuh>
#include <queue>

#include "puzzle/GPU-solver.cuh"
#include "puzzle/GPU-kernel.cuh"

using namespace mgpu;

int div_up(int x, int y) { return (x-1) / y + 1; }

struct DeviceData {
    // store the structure of the grid graph
    MGPU_MEM(uint8_t) graph;

    // store open list + close list
    MGPU_MEM(node_t) nodes;
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

    // current shortest distance (a float)
    MGPU_MEM(uint32_t) optimalDistance;
    // store the result return by the GPU
    MGPU_MEM(heap_t) optimalNodes;
    // store the size for optimalNodes
    MGPU_MEM(int) optimalNodesSize;

    MGPU_MEM(uint32_t) lastAddr;
    MGPU_MEM(uint32_t) answerList;
    MGPU_MEM(int) answerSize;

    ContextPtr context;
};


GPUPathwaySolver::GPUPathwaySolver(Pathway *pathway)
    : p(pathway)
{
    d = new DeviceData();
}

GPUPathwaySolver::~GPUPathwaySolver()
{
    // vector<node_t> nodes;
    // vector<uint32_t> hash;
    // d->nodes->ToHost(nodes, d->nodeSize->Value());
    // d->hash->ToHost(hash, p->size());
    // for (;;) {
    //     cout << "(x, y): ";
    //     int x, y;
    //     int px, py;
    //     cin >> x >> y;
    //     int nodeID = p->toID(x, y);
    //     int hashValue = hash[nodeID];
    //     int prevID = nodes[nodes[hashValue].prev].nodeID;
    //     p->toXY(prevID, &px, &py);
    //     std::cout << "fValue: " << nodes[hashValue].fValue << endl
    //               << "gValue: " << nodes[hashValue].gValue << endl
    //               << "prev: " << px << ", " << py << endl << endl;;
    // }
    delete d;
}

void GPUPathwaySolver::initialize()
{
    cudaDeviceSynchronize();
    cudaDeviceReset();

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
            pd.fetchDatabase(database[i].data());

            index[i].resize(tracked[i].size());
            multiple[i].reserve(tracked[i].size() + 1);

            for (int j = 0; j < (int)tracked[i].size(); ++j) {
                mapTracked[tracked[i][j]] = make_pair(i, j);
                multiple[i].push_back(N*N - j);
            }
            multiple[i].push_back(1);

            for (int j = tracked[i].size()-2; j >= 0; --j)
                multiple[i][j] *= multiple[i][j+1];

            for (int j = 0; j <= tracked[i].size(); ++j)
                cout << multiple[i][j] << " ";
            cout << endl;
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


    d->context = CreateCudaDevice(vm_options["ordinal"].as<int>());

    initializeCUDAConstantMemory(
        p->height(), p->width(), p->ex(), p->ey(),
        (uint32_t)p->toID(p->ex(), p->ey()));

    d->graph = d->context->Malloc<uint8_t>(p->graph(), p->size());

    d->nodes = d->context->Malloc<node_t>(NODE_LIST_SIZE);
    d->nodeSize = d->context->Fill<int>(1, 1);

    d->hash = d->context->Fill<uint32_t>(p->size(), UINT32_MAX);

    d->openList = d->context->Malloc<heap_t>(OPEN_LIST_SIZE);
    d->heapSize = d->context->Fill<int>(NUM_TOTAL, 0);
    d->heapBeginIndex = d->context->Fill<int>(1, 0);

    d->sortList = d->context->Malloc<sort_t>(NUM_VALUE * 8);
    d->prevList = d->context->Malloc<uint32_t>(NUM_VALUE * 8);
    d->sortList2 = d->context->Malloc<sort_t>(NUM_VALUE * 8);
    d->prevList2 = d->context->Malloc<uint32_t>(NUM_VALUE * 8);
    d->sortListSize = d->context->Fill<int>(1, 0);
    d->sortListSize2 = d->context->Fill<int>(1, 0);

    d->heapInsertList = d->context->Malloc<heap_t>(NUM_VALUE * 8);
    d->heapInsertSize = d->context->Fill<int>(1, 0);

    d->optimalDistance = d->context->Fill<uint32_t>(1, UINT32_MAX);
    d->optimalNodes = d->context->Malloc<heap_t>(NUM_TOTAL);
    d->optimalNodesSize = d->context->Fill<int>(1, 0);

    d->lastAddr = d->context->Malloc<uint32_t>(1);
    d->answerList = d->context->Malloc<uint32_t>(ANSWER_LIST_SIZE);
    d->answerSize = d->context->Fill<int>(1, 0);

    kInitialize<<<1, 1>>>(
        *d->nodes,
        *d->hash,
        *d->openList,
        *d->heapSize,
        p->sx(),
        p->sy()
    );
    dout << "\t\tGPU Initialization finishes" << endl;
}

bool GPUPathwaySolver::solve()
{
    std::priority_queue< heap_t, vector<heap_t>, std::greater<heap_t> > pq;

    for (int round = 0; ;++round) {
        if (DEBUG_CONDITION) {
            vector<int> heapSize;
            d->heapSize->ToHost(heapSize, NUM_TOTAL);
            printf("\t\t\t Heapsize: %d of %d\n", heapSize[0], HEAP_CAPACITY);
        }

        // printf("\t\tRound %d\n", round); fflush(stdout);
        dprintf("\t\tRound %d: kExtractExpand\n", round);
        kExtractExpand<
            NUM_BLOCK, NUM_THREAD, VALUE_PER_THREAD, HEAP_CAPACITY> <<<
            NUM_BLOCK, NUM_THREAD>>>(
                *d->nodes,

                *d->graph,

                *d->openList,
                *d->heapSize,

                *d->optimalDistance,
                *d->optimalNodes,
                *d->optimalNodesSize,

                *d->sortList,
                *d->prevList,
                *d->sortListSize,

                // reset them BTW
                *d->heapBeginIndex,
                *d->heapInsertSize
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

            uint32_t optimalDistance = d->optimalDistance->Value();
            dprintf("\t\tRound %d: Fetch optimalDistance: %.2f\n", round, reverseFlipFloat(optimalDistance));

            for (size_t i = 0; i != optimalNodes.size(); ++i) {
                dprintf("\t\t\t optimalNodes[%d]: %.3f\n", (int)i, optimalNodes[i].fValue);
                pq.push(optimalNodes[i]);
            }

            dprintf("\t\t\t pq.top(): %.3f\n", pq.top().fValue);
            if (flipFloat(pq.top().fValue) <= optimalDistance) {
                printf("\t\t\t Number of nodes expanded: %d\n", d->nodeSize->Value());
                m_optimalNodeAddr = pq.top().addr;
                m_optimalDistance = pq.top().fValue;
                dprintf("\t\t\t Optimal nodes address: %d\n", m_optimalNodeAddr);
                return true;
            }
        }

        dprintf("\t\tRound %d: Fetch sortListSize: ", round);
        int sortListSize = d->sortListSize->Value();
        dprintf("%d\n", sortListSize);
        // if (round % 2000 == 0) {
        //     printf("\t\tRound %d: Fetch sortListSize: %d\n", round, sortListSize);
        // }
        if (sortListSize == 0)
            return false;

        dprintf("\t\tRound %d: MergesortPairs\n", round);
        MergesortPairs(
            d->sortList->get(),
            d->prevList->get(),
            sortListSize,
            *d->context
        );

        dprintf("\t\tRound %d: kAssign\n", round);
        kAssign<NUM_THREAD><<<
            div_up(sortListSize, NUM_THREAD), NUM_THREAD>>> (
                *d->sortList,
                *d->prevList,
                sortListSize,

                *d->sortList2,
                *d->prevList2,
                *d->sortListSize2
            );
#ifdef KERNEL_LOG
        cudaDeviceSynchronize();
#endif

        dprintf("\t\tRound %d: Fetch sortListSize2: ", round);
        int sortListSize2 = d->sortListSize2->Value();
        dprintf("%d\n", sortListSize2);
        // if (round % 2000 == 0) {
        //     printf("\t\tRound %d: Fetch sortListSize2: %d\n", round, sortListSize2);
        // }

        dprintf("\t\tRound %d: kDeduplicate\n", round);
        // printf("\t\tRound %d: nodeSize: %d\n", round, d->nodeSize->Value());
        kDeduplicate<NUM_THREAD> <<<
            div_up(sortListSize2, NUM_THREAD), NUM_THREAD>>> (
                *d->nodes,
                *d->nodeSize,

                *d->hash,

                *d->sortList2,
                *d->prevList2,
                sortListSize2,

                *d->heapInsertList,
                *d->heapInsertSize
            );
        // printf("\t\tRound %d: nodeSize: %d\n", round, d->nodeSize->Value());
#ifdef KERNEL_LOG
        cudaDeviceSynchronize();
#endif

        dprintf("\t\tRound %d: kHeapInsert\n", round);
        kHeapInsert<
            NUM_BLOCK, NUM_THREAD, HEAP_CAPACITY> <<<
            NUM_BLOCK, NUM_THREAD>>> (
                *d->openList,
                *d->heapSize,
                *d->heapBeginIndex,

                *d->heapInsertList,
                *d->heapInsertSize,

                // reset them BTW
                *d->sortListSize,
                *d->sortListSize2,
                *d->optimalDistance,
                *d->optimalNodesSize
            );
#ifdef KERNEL_LOG
        cudaDeviceSynchronize();
#endif
        dprintf("\t\tRound %d: Finished\n\n", round);
    }
}

void GPUPathwaySolver::getSolution(float *optimal, vector<int> *pathList)
{
    d->lastAddr->FromHost(&m_optimalNodeAddr, 1);
    kFetchAnswer<<<1, 1>>>(
        *d->nodes,

        *d->lastAddr,

        *d->answerList,
        *d->answerSize
    );

    int answerSize = d->answerSize->Value();

    vector<uint32_t> answerList;
    d->answerList->ToHost(answerList, answerSize);

    *optimal = m_optimalDistance;
    pathList->clear();
    pathList->reserve(answerSize);
    for (int i = answerSize-1; i >= 0; --i) {
        pathList->push_back((int)answerList[i]);
    }

}

bool GPUPathwaySolver::isPrime(uint32_t number)
{
    uint32_t upper = sqrt(number) + 1;
    assert(upper < number);

    for (uint32_t i = 2; i != upper; ++i)
        if (number % i == 0)
            return false;
    return true;
}

vector<uint32_t> GPUPathwaySolver::genRandomPrime(uint32_t maximum, int count)
{
    vector<uint32_t> result;
    int prepare = 3 * count;

    uint32_t now = maximum;
    while (prepare) {
        if (isPrime(now))
            result.push_back(now);
        now--;
    }

    std::random_shuffle(result.begin(), result.end());
    result.erase(result.begin() + count, result.end());

    for (int i = 0; i < count; ++i)
        dout << result[i] << " ";
    dout << endl;

    return result;
}
