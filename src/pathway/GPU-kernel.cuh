#ifndef __GPU_KERNEL_CUH_IUGANILK
#define __GPU_KERNEL_CUH_IUGANILK

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <moderngpu.cuh>

#include "utils.hpp"

// Suppose we only use x dimension
#define THREAD_ID (threadIdx.x)
#define GLOBAL_ID (THREAD_ID + NT * blockIdx.x)
#define BLOCK_ID  (blockIdx.x)

using namespace mgpu;

struct heap_t {
    float fValue;
    uint32_t addr;
};

CUDA_FUNC bool operator<(const heap_t &a, const heap_t &b)
{
    return a.fValue < b.fValue;
}

struct node_t {
    int nodeID;
    uint32_t prev;
    float fValue;
    float gValue;
};

struct sort_t {
    int nodeID;
    float gValue;
};

CUDA_FUNC bool operator<(const sort_t &a, const sort_t &b)
{
    if (a.nodeID != b.nodeID)
        return a.nodeID < b.nodeID;
    return a.gValue < b.gValue;
}

struct hash_t {
    int nodeID;
    uint32_t addr;
};

inline CUDA_FUNC uint32_t flipFloat(float fl)
{
    union {
        float fl;
        uint32_t  u;
    } un;
    un.fl = fl;
    return un.u ^ ((un.u >> 31) | 0x80000000);
}

inline CUDA_FUNC float reverseFlipFloat(uint32_t u)
{
    union {
        float f;
        uint32_t u;
    } un;
    un.u = u ^ ((~u >> 31) | 0x80000000);
    return un.f;
}

__constant__ int d_height;
__constant__ int d_width;
__constant__ int d_targetX;
__constant__ int d_targetY;
__constant__ int d_targetID;

inline CUDA_FUNC void idToXY(int nodeID, int *x, int *y)
{
    *x = nodeID / d_width;
    *y = nodeID % d_width;
}

inline CUDA_FUNC int xyToID(int x, int y)
{
    return x * d_width + y;
}

inline CUDA_FUNC float computeHValue(int x, int y)
{
}

inline CUDA_FUNC float inrange(int x, int y)
{
    return 0 <= x && x < d_height && 0 <= y && y < d_width;
}

cudaError_t initializeCUDAConstantMemory(
    int height,
    int width,
    int targetX,
    int targetY,
    int targetID
)
{
    ret = 0;
    ret |= cudaMemcpyToSymbol(d_height, &height, sizeof(int));
    ret |= cudaMemcpyToSymbol(d_width, &width, sizeof(int));
    ret |= cudaMemcpyToSymbol(d_targetX, &targetX, sizeof(int));
    ret |= cudaMemcpyToSymbol(d_targetY, &targetY, sizeof(int));
    ret |= cudaMemcpyToSymbol(d_targetID, &targetID, sizeof(int));
    return ret;
}

template<int NB, int NT, int VT, int HEAP_CAPACITY>
CUDA_KERNEL void kExtractExpand(
    // global nodes
    node_t g_nodes[],

    uint8_t g_graph[],

    // open list
    heap_t g_openList[],
    int g_heapSize[],

    // solution
    uint32_t *g_optimalDistance,
    node_t g_optimalNodes[],
    int *g_optimalNodesSize,

    // output buffer
    sort_t g_sortList[],
    uint32_t g_prevList[],
    int *g_sortListSize,
)
{

    __shared__ uint32_t s_optimalDistance;
    __shared__ int s_sortListSize;
    __shared__ int s_sortListIndex;

    int gid = GLOBAL_ID;
    int tid = THREAD_ID;
    if (tid == 0) {
        s_optimalDistance = UINT32_MAX;
        s_sortListSize = 0;
        s_sortListIndex = 0;
    }

    heap_t *heap = g_openList + HEAP_CAPACITY * gid - 1;

    heap_t extracted[VT];
    int popCount = 0;
    int heapSize = g_heapSize[gid];

#pragma unroll
    for (int k = 0; k < VT; ++k) {
        if (heapSize == 0)
            break;

        extracted[popCount] = heap[1];
        atomicMin(&s_optimalDistance, flipFloat(extracted[popCount].fValue));
        popCount++;

        heap_t nowValue = heap[heapSize--];

        int now = 1;
        int next;
        while ((next = now*2) <= heapSize) {
            heap_t nextValue = heap[next];
            heap_t nextValue2 = heap[next+1];
            bool inc = (next+1 <= heapSize) && (nextValue2 < nextValue);
            if (inc) {
                ++next;
                nextValue = nextValue2;
            }

            if (nextValue < nowValue) {
                heap[now] = nextValue;
                now = next;
            } else
                break;
        }
        heap[now] = nowValue;

    }
    g_heapSize[gid] = heapSize;

    int sortListCount = 0;
    sort_t sortList[VT*8];
    int prevList[VT*8];

    const int DX[8] = { 1,  1, -1, -1,  1, -1,  0,  0 };
    const int DY[8] = { 1, -1,  1, -1,  0,  0,  1, -1 };
    const float COST[8] = {
        1.4142135623731, 1.4142135623731,
        1.4142135623731, 1.4142135623731,
        1, 1, 1, 1};

    for (int k = 0; k < popCount; ++k) {
        node_t node = g_nodes[extracted[k].addr];
        if (extracted[k].fValue != node.fValue)
            continue;

        if (node.nodeID == targetID) {
            int index = atomicAdd(g_optimalNodesSize, 1);
            g_optimalNodes[index] = node;
        }

        int x, y;
        idToXY(node.nodeID, &x, &y);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (~g_graph[node.nodeID] & 1 << i)
                continue;

            int nx = x + DX[i];
            int ny = y + DY[i];
            if (inrange(nx, ny)) {
                sortList[sortListCount].nodeID = xyToID(nx, ny);
                sortList[sortListCount].gValue = node.gValue + COST[i];
                prevList[sortListCount] = node.nodeID;
                ++sortListCount;
            }
        }
    }

    int sortListIndex = atomicAdd(&s_sortListSize, sortListCount);
    __syncthreads();
    if (tid == 0) {
        s_sortListIndex = atomicAdd(g_sortListSize, s_sortListSize);
    }
    __syncthreads();
    sortListIndex += s_sortListIndex;

    for (int k = 0; k < sortListCount; ++k) {
        g_sortList[sortListIndex + k] = sortList[k];
        g_prevList[sortListIndex + k] = prevList[k];
    }

    if (tid == 0)
        atomicMin(g_optimalDistance, s_optimalDistance);
}

template<int NB, int NT, int VT, int NH, int HASHTABLE_SIZE>
CUDA_KERNEL void kDeduplicate(
    // global nodes
    node_t g_nodes[],

    // hash table
    hash_t g_hash[],

    // modules used in sub hash table
    uint32_t modules[],

    // output buffer
    sort_t g_sortList[],
    uint32_t g_prevList[],
    int sortListSize
)
{
    int bid = BLOCK_ID;
    int tid = THREAD_ID;
    int gid = GLOBAL_ID;

    hash_t *hash = g_hash + HASH_SIZE * bid;

    __shared__ sort_t s_sortList[NT*VT+1];
    __shared__ uint32_t s_prevList[NT*VT+1];

    sort_t sortList[VT];
    uint32_t prevList[VT];

    DeviceGlobalToThread<NT, VT>(sortListSize, g_sortList, gid, sortList);
    DeviceGlobalToThread<NT, VT>(sortListSize, g_prevList, gid, prevList);

    CTAMergesort<NT, VT, false, true>(
        sortList, prevList, s_sortList, s_prevList,
        sortListSize, tid, mgpu::less<sort_t>());

    typedef CTAScan<NT> S;
    __shared__ S::Storage scanStorage;

#pragma unroll
    for (int k = 0; k < VT; ++k) {
        int base = k * NT;
        int pos = base + tid;
        bool working = pos < sortListSize;

        int keep;
        int index;
        int maxIndex;
        sort_t sortElement;
        uint32_t prevElement;
        if (working) {
            keep = (pos == 0 ||
                    sortList[pos].nodeID != sortList[pos-1].nodeID]);
            index;
            maxIndex = S::Scan(tid, keep, scanStorage, &index);

            --index;
            sortElement = s_sortList[pos];
            prevElement = s_prevList[pos];
        }
        __syncthreads();

        if (working) {
            if (keep) {
                s_sortList[base + index] = sortElement;
                s_prevList[base + index] = prevElement;
            }
            working = tid < maxIndex;
        }
        __syncthreads();

        // lookup it in cuckoo hash
        if (working) {
            bool success = false;

            int nodeID = s_sortList[pos].nodeID;
            float gValue = s_sortList[pos].gValue;
            float fValue = gValue + computeHValue
            uint32_t prev = s_prevList[pos].prev;

            hash_t *subHash;

            subHash = hash;
#pragma unroll
            for (int i = 0; i < NH; ++i) {
                subHash += modules[i];
            }
        }
    }
}

template<int NB, int NT, int VT>
CUDA_KERNEL void kInsert(
)
{
}

template<int NB, int NT, int VT>
CUDA_KERNEL void kRebuild(
)
{
}

#endif /* end of include guard: __GPU_KERNEL_CUH_IUGANILK */
