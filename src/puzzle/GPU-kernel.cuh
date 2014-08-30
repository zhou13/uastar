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

__host__ __device__ bool operator<(const heap_t &a, const heap_t &b)
{
    return a.fValue < b.fValue;
}

struct node_t {
    uint32_t prev;
    float fValue;
    float gValue;
    uint32_t nodeID;
};

struct sort_t {
    uint32_t nodeID;
    float gValue;
};

__host__ __device__ bool operator<(const sort_t &a, const sort_t &b)
{
    if (a.nodeID != b.nodeID)
        return a.nodeID < b.nodeID;
    return a.gValue < b.gValue;
}

struct hash_t {
    uint32_t nodeID;
    uint32_t addr;
};

inline __host__ __device__ uint32_t flipFloat(float fl)
{
    union {
        float fl;
        uint32_t  u;
    } un;
    un.fl = fl;
    return un.u ^ ((un.u >> 31) | 0x80000000);
}

inline __host__ __device__ float reverseFlipFloat(uint32_t u)
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
__constant__ uint32_t d_targetID;
__constant__ uint32_t d_modules[10];

inline __device__ void idToXY(uint32_t nodeID, int *x, int *y)
{
    *x = nodeID / d_width;
    *y = nodeID % d_width;
}

inline __device__ int xyToID(int x, int y)
{
    return x * d_width + y;
}

inline __device__ float computeHValue(int x, int y)
{
    int dx = abs(d_targetX - x);
    int dy = abs(d_targetY - y);
    return min(dx, dy)*SQRT2 + abs(dx-dy);
}

inline __device__ float computeHValue(uint32_t nodeID)
{
    int x, y;
    idToXY(nodeID, &x, &y);
    return computeHValue(x, y);
}

inline __device__ float inrange(int x, int y)
{
    return 0 <= x && x < d_height && 0 <= y && y < d_width;
}

inline cudaError_t initializeCUDAConstantMemory(
    int height,
    int width,
    int targetX,
    int targetY,
    uint32_t targetID
)
{
    cudaError_t ret = cudaSuccess;
    ret = cudaMemcpyToSymbol(d_height, &height, sizeof(int));
    ret = cudaMemcpyToSymbol(d_width, &width, sizeof(int));
    ret = cudaMemcpyToSymbol(d_targetX, &targetX, sizeof(int));
    ret = cudaMemcpyToSymbol(d_targetY, &targetY, sizeof(int));
    ret = cudaMemcpyToSymbol(d_targetID, &targetID, sizeof(uint32_t));
    return ret;
}

inline cudaError_t updateModules(const vector<uint32_t> &mvec)
{
    return cudaMemcpyToSymbol(
        d_modules, mvec.data(), sizeof(uint32_t) * mvec.size())
}


// NB: number of CUDA block
// NT: number of CUDA thread per CUDA block
// VT: value handled per thread
// M0/M:  nodeID will be stored in g_hash[nodeID % M0 % M]
template<int NB, int NT, int VT, int HEAP_CAPACITY, uint32_t M0, uint32_t M>
__global__ void kExtractExpand(
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
    int *g_sortListSize
)
{
    __shared__ uint32_t s_optimalDistance;
    __shared__ int s_sortListSize;
    __shared__ int s_sortListIndex;
    __shared__ uint32_t s_histogram[NS];

    typedef CTAScan<M> S;
    __shared__ typename S::Storage scanStorage;

    int gid = GLOBAL_ID;
    int tid = THREAD_ID;
    if (tid == 0) {
        s_optimalDistance = UINT32_MAX;
        s_sortListSize = 0;
        s_sortListIndex = 0;
    }
    if (tid < M) {
        s_histogram[tid] = 0;
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

        if (node.nodeID == d_targetID) {
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
                uint32_t nodeID = xyToID(nx, ny);
                sortList[sortListCount].nodeID = nodeID;
                sortList[sortListCount].gValue = node.gValue + COST[i];
                prevList[sortListCount] = node.nodeID;
                atomicAdd(s_histogram + nodeID % M0 % M, 1);
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
    if (tid < M) {
        int position;
        S::Scan(tid, s_histogram[tid], scanStorage, &position);
        atomicAdd(g_position + tid, position);
    }
    if (tid == 0)
        atomicMin(g_optimalDistance, s_optimalDistance);
}

// Assume g_sortList is sorted
template<int NB, int NT, uint32_t M0, uint32_t M>
__global__ void kAssign(
    int g_position[],

    sort_t g_sortList[],
    uint32_t g_prevList[],

    sort_t g_sortList2[],
    uint32_t g_prevList2[],

    int sortListSize
)
{
    __shared__ int s_position[M];
    __shared__ uint32_t s_nodeIDList[NT+1];

    int tid = THREAD_ID;
    int gid = GLOBAL_ID;
    int bid = BLOCK_ID;

    if (tid < M) {
        s_position[tid] = 0;
    }

    bool working = false;
    sort_t sort;
    uint32_t prev;
    uint32_t category;

    if (tid == 0 && gid != 0)
        s_nodeIDList[0] = g_sortList[gid - 1];

    if (gid < sortListSize) {
        working = true;
        sort = g_sortList[gid];
        prev = g_prevList[gid];
        s_nodeIDList[1 + tid] = g_sortList[gid].nodeID;
    }
    __syncthreads();

    working &&= (gid == 0 || s_nodeIDList[tid] != s_nodeIDList[tid+1]);

    if (working) {
        category = sort.nodeID % M0 % M;
        atomicAdd(s_position + category, 1);
    }
    __syncthreads();
    if (tid < M) {
        int position = s_position[tid];
        int end = atomicSub(g_position[tid], position);
        s_position[tid] = end - position;
    }
    __syncthreads();

    if (working) {
        int index = atomicAdd(s_position[category], 1);

        g_sortList2[index] = sort;
        g_prevList2[index] = prev;
    }
}

template<int NB, int NT, int NH, int HASH_SIZE, int MAX_RETRY>
__global__ void kDeduplicate(
    // global nodes
    node_t g_nodes[],
    int *g_nodeSize,

    // hash table
    hash_t g_hash[],

    // output buffer
    sort_t g_sortList[],
    uint32_t g_prevList[],
    int sortListSize,

    bool *restart,
)
{
    int bid = BLOCK_ID;
    int tid = THREAD_ID;
    int gid = GLOBAL_ID;

    int begin = g_position[bid];
    int end = (bid == NB ? sortListSize : g_position[bid+1]);

    hash_t *hash = g_hash + HASH_SIZE * bid;

    __shared__ int s_insertNodeCount;
    __shared__ uint32_t s_insertNodes[NT];


    int base = begin;
    while (base < end) {
        int index = base + tid;
        bool working = base < size;

        bool found = false;
        bool insert = false;

        sort_t sort;
        uint32_t prev;
        uint32_t nodeID;
        float gValue;
        float fValue;

        uint32_t hashValue[NH];

        if (tid == 0)
            s_insertNodeCount = 0;

        if (working) {
            sort = g_sortList[index];
            prev = g_prevList[index];
            nodeID = sort.nodeID;
            gValue = sort.gValue;
            fValue = gValue + computeHValue(nodeID);

            hash_t *subHash = hash;
#pragma unroll
            for (int i = 0; i < NH; ++i) {
                uint32_t hashValue[i] = nodeID % d_modules[i];
                hash_t hashNode = subHash[hashValue[i]];
                if (hashNode.nodeID == nodeID) {
                    found = true;

                    node_t node = g_nodes[hashNode.addr];
                    if (fValue < node.fValue) {
                        insert = true;
                        g_nodes[hashNode.addr].fValue = fValue;
                    }
                    break;
                }
                subHash += modules[i];
            }
        }

        bool success = false;
#pragma unroll
        for (int k = 0; k < MAX_RETRY; ++k) {
            hash_t *subHash = hash;
#pragma unroll
            for (int i = 0; i < NH; ++i) {
                if (working && !success) {
                    if (subHash[hashValue[i]].nodeID == UINT32_MAX) {
                        subHash[hashValue[i]].nodeID = nodeID;
                    }
                }

                __syncthreads();

                if (working && !success) {
                    if (subHash[hashValue[i]].nodeID == nodeID) {
                        success = true;
                    }
                }
                subHash += modules[i];
            }
        }

        if (insert) {
            int index = atomicAdd(&s_insertNodeCount, 1);
        }

        __syncthreads();

        base += VT;
    }
}

template<int NB, int NT, int VT>
__global__ void kHeapInsert(
)
{
}

#endif /* end of include guard: __GPU_KERNEL_CUH_IUGANILK */
