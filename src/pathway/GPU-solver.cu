#define NO_CPP11

#include <moderngpu.cuh>

#include "pathway/GPU-solver.hpp"
#include "pathway/GPU-kernel.cuh"


using namespace mgpu;

struct DeviceData {
    // store the structure of the grid graph
    MGPU_MEM(uint8_t) graph;

    // store open list + close list
    MGPU_MEM(node_t) nodes;

    // hash table for `nodes'
    MGPU_MEM(hash_t) hashTable;
    // define the modules of sub hash table
    MGPU_MEM(uint32_t) modules;

    // store open list
    MGPU_MEM(heap_t) openList;
    // store the size for each heap
    MGPU_MEM(int) heapSize;

    // element waiting to be sorted
    MGPU_MEM(sort_t) sortList;
    // value for sortList, representing the its parents
    MGPU_MEM(uint32_t) prevList;
    // size of the preceding array
    MGPU_MEM(int) sortListSize;

    // current shortest distance (a float)
    MGPU_MEM(uint32_t) optimalDistance;
    // store the result return by the GPU
    MGPU_MEM(node_t) optimalNodes;
    // store the size for optimalNodes
    MGPU_MEM(int) optimalNodesSize;

    ContextPtr context;
};


GPUPathwaySolver::GPUPathwaySolver(Pathway *pathway)
    : p(pathway)
{
    d = new DeviceData();
}

GPUPathwaySolver::~GPUPathwaySolver()
{
    delete d;
}

void GPUPathwaySolver::initialize()
{
    d->context = CreateCudaDevice(vm_options["ordinal"].as<int>());

    d->graph = d->context->Malloc<uint8_t>(p->graph(), p->size());

    d->openList = d->context->Malloc<heap_t>(OPEN_LIST_SIZE);

    d->nodes = d->context->Malloc<node_t>(NODE_LIST_SIZE);

    d->heapSize = d->context->Fill<int32_t>(NUM_TOTAL, 0);

    d->optimalDistance = d->context->Fill<uint32_t>(1, UINT32_MAX);
}

bool GPUPathwaySolver::solve(float *optimal, vector<vec2> *solution)
{
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

            p->toID(p->ex(), p->ey()),
            p->height(),
            p->width()
        );
    // kDeduplicate<
    //     NUM_BLOCK, NUM_THREAD, VALUE_PER_THREAD
    return false;
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

