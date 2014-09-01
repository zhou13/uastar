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

#define cudaAssert(X) \
    if ( !(X) ) { \
        printf( "Thread %d:%d failed assert at %s:%d!\n", \
                blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); \
        return; \
    }
// #define KERNEL_LOG

using namespace mgpu;

struct heap_t {
    float fValue;
    uint32_t addr;
};

__host__ __device__ bool operator<(const heap_t &a, const heap_t &b)
{
    return a.fValue < b.fValue;
}
__host__ __device__ bool operator>(const heap_t &a, const heap_t &b)
{
    return a.fValue > b.fValue;
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
        d_modules, mvec.data(), sizeof(uint32_t) * mvec.size());
}


__global__ void kInitialize(
    node_t g_nodes[],
    uint32_t g_hash[],
    heap_t g_openList[],
    int g_heapSize[],
    int startX,
    int startY
)
{
    node_t node;
    node.fValue = computeHValue(startX, startY);
    node.gValue = 0;
    node.prev = UINT32_MAX;
    node.nodeID = xyToID(startX, startY);

    heap_t heap;
    heap.fValue = node.fValue;
    heap.addr = 0;

    g_nodes[0] = node;
    g_openList[0] = heap;
    g_heapSize[0] = 1;
    g_hash[node.nodeID] = 0;
}

// NB: number of CUDA block
// NT: number of CUDA thread per CUDA block
// VT: value handled per thread
template<int NB, int NT, int VT, int HEAP_CAPACITY>
__global__ void kExtractExpand(
    // global nodes
    node_t g_nodes[],

    uint8_t g_graph[],

    // open list
    heap_t g_openList[],
    int g_heapSize[],

    // solution
    uint32_t *g_optimalDistance,
    heap_t g_optimalNodes[],
    int *g_optimalNodesSize,

    // output buffer
    sort_t g_sortList[],
    uint32_t g_prevList[],
    int *g_sortListSize,

    // cleanup
    int *g_heapBeginIndex,
    int *g_heapInsertSize
)
{
    __shared__ uint32_t s_optimalDistance;
    __shared__ int s_sortListSize;
    __shared__ int s_sortListBase;

    int gid = GLOBAL_ID;
    int tid = THREAD_ID;
    if (tid == 0) {
        s_optimalDistance = UINT32_MAX;
        s_sortListSize = 0;
        s_sortListBase = 0;
    }

    __syncthreads();

    heap_t *heap = g_openList + HEAP_CAPACITY * gid - 1;

    heap_t extracted[VT];
    int popCount = 0;
    int heapSize = g_heapSize[gid];

#pragma unroll
    for (int k = 0; k < VT; ++k) {
        if (heapSize == 0)
            break;

        extracted[k] = heap[1];
        popCount++;

#ifdef KERNEL_LOG
        int x, y;
        idToXY(g_nodes[extracted[k].addr].nodeID, &x, &y);
        printf("\t\t\t[%d]: Extract (%d, %d){%.2f} in [%d]\n",
               gid, x, y, extracted[k].fValue, extracted[k].addr);
#endif

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
    bool valid[VT*8];

    const int DX[8] = { 1,  1, -1, -1,  1, -1,  0,  0 };
    const int DY[8] = { 1, -1,  1, -1,  0,  0,  1, -1 };
    const float COST[8] = { SQRT2, SQRT2, SQRT2, SQRT2, 1, 1, 1, 1 };

#pragma unroll
    for (int k = 0; k < VT; ++k) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            valid[k*8 + i] = false;

        if (k >= popCount)
            continue;
        atomicMin(&s_optimalDistance, flipFloat(extracted[k].fValue));
        node_t node = g_nodes[extracted[k].addr];
        if (extracted[k].fValue != node.fValue)
            continue;

        if (node.nodeID == d_targetID) {
            int index = atomicAdd(g_optimalNodesSize, 1);
            g_optimalNodes[index] = extracted[k];
#ifdef KERNEL_LOG
            printf("\t\t\t Saved answer {%.3f}\n", extracted[k].fValue);
#endif
            continue;
        }

        int x, y;
        idToXY(node.nodeID, &x, &y);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (~g_graph[node.nodeID] & (1 << i))
                continue;

            int nx = x + DX[i];
            int ny = y + DY[i];
            int index = k*8 + i;
            if (inrange(nx, ny)) {
                uint32_t nodeID = xyToID(nx, ny);
#ifdef KERNEL_LOG
                int px, py;
                idToXY(node.nodeID, &px, &py);
                printf("\t\t\t[%d]: Expand (%d, %d) from (%d, %d)\n",
                       gid, nx, ny, px, py);
#endif
                sortList[index].nodeID = nodeID;
                sortList[index].gValue = node.gValue + COST[i];
                prevList[index] = extracted[k].addr;
                valid[index] = true;
                ++sortListCount;
            }
        }
    }

    int sortListIndex = atomicAdd(&s_sortListSize, sortListCount);
    __syncthreads();
    if (tid == 0) {
        s_sortListBase = atomicAdd(g_sortListSize, s_sortListSize);
    }
    __syncthreads();
    sortListIndex += s_sortListBase;

#pragma unroll
    for (int k = 0; k < VT*8; ++k)
        if (valid[k]) {
            g_sortList[sortListIndex] = sortList[k];
            g_prevList[sortListIndex] = prevList[k];
            sortListIndex++;
        }
    if (tid == 0)
        atomicMin(g_optimalDistance, s_optimalDistance);
    if (gid == 0) {
        int newHeapBeginIndex = *g_heapBeginIndex + *g_heapInsertSize;

        *g_heapBeginIndex = newHeapBeginIndex % (NB*NT);
        *g_heapInsertSize = 0;
    }
}

// Assume g_sortList is sorted
template<int NT>
__global__ void kAssign(
    sort_t g_sortList[],
    uint32_t g_prevList[],
    int sortListSize,

    sort_t g_sortList2[],
    uint32_t g_prevList2[],
    int *g_sortListSize2
)
{
    __shared__ uint32_t s_nodeIDList[NT+1];
    __shared__ uint32_t s_sortListCount2;
    __shared__ uint32_t s_sortListBase2;

    int tid = THREAD_ID;
    int gid = GLOBAL_ID;

    bool working = false;
    sort_t sort;
    uint32_t prev;

    if (tid == 0)
        s_sortListCount2 = 0;

    if (tid == 0 && gid != 0)
        s_nodeIDList[0] = g_sortList[gid - 1].nodeID;

    if (gid < sortListSize) {
        working = true;
        sort = g_sortList[gid];
        prev = g_prevList[gid];
        s_nodeIDList[tid+1] = sort.nodeID;
    }
    __syncthreads();

    working &= (gid == 0 || s_nodeIDList[tid] != s_nodeIDList[tid+1]);

    int index;
    if (working) {
        index = atomicAdd(&s_sortListCount2, 1);
    }

    __syncthreads();
    if (tid == 0) {
         s_sortListBase2 = atomicAdd(g_sortListSize2, s_sortListCount2);
    }
    __syncthreads();

    if (working) {
        g_sortList2[s_sortListBase2 + index] = sort;
        g_prevList2[s_sortListBase2 + index] = prev;

#ifdef KERNEL_LOG
        int x, y;
        idToXY(sort.nodeID, &x, &y);
        printf("\t\t\t[%d]: Assign (%d %d){%.2f} from %d\n",
               gid, x, y, sort.gValue, prev);
#endif
    }
}

template<int NT>
__global__ void kDeduplicate(
    // global nodes
    node_t g_nodes[],
    int *g_nodeSize,

    // hash table
    uint32_t g_hash[],

    sort_t g_sortList[],
    uint32_t g_prevList[],
    int sortListSize,

    heap_t g_heapInsertList[],
    int *g_heapInsertSize
)
{
    int tid = THREAD_ID;
    int gid = GLOBAL_ID;
    bool working = gid < sortListSize;

    __shared__ int s_nodeInsertCount;
    __shared__ int s_nodeInsertBase;

    __shared__ int s_heapInsertCount;
    __shared__ int s_heapInsertBase;

    if (tid == 0) {
        s_nodeInsertCount = 0;
        s_heapInsertCount = 0;
    }
    __syncthreads();

    node_t node;
    bool insert = true;
    bool found = true;
    uint32_t nodeIndex;
    uint32_t heapIndex;
    uint32_t addr;

    if (working) {
        node.nodeID = g_sortList[gid].nodeID;
        node.gValue = g_sortList[gid].gValue;
        node.prev   = g_prevList[gid];
        node.fValue = node.gValue + computeHValue(node.nodeID);

        // cudaAssert((int)node.nodeID >= 0);
        addr = g_hash[node.nodeID];
        found = (addr != UINT32_MAX);

        if (found) {
            if (node.fValue < g_nodes[addr].fValue) {
                g_nodes[addr] = node;
            } else {
                insert = false;
            }
        }

        if (!found) {
            nodeIndex = atomicAdd(&s_nodeInsertCount, 1);
        }
        if (insert) {
            heapIndex = atomicAdd(&s_heapInsertCount, 1);
        }
    }

    __syncthreads();
    if (tid == 0) {
        s_nodeInsertBase = atomicAdd(g_nodeSize, s_nodeInsertCount);
        s_heapInsertBase = atomicAdd(g_heapInsertSize, s_heapInsertCount);
    }
    __syncthreads();

    if (working && !found) {
        addr = s_nodeInsertBase + nodeIndex;
#ifdef KERNEL_LOG
        int x, y;
        idToXY(node.nodeID, &x, &y);
        printf("\t\t\t[%d]: Store (%d, %d) to [%d]\n", gid, x, y, addr);
#endif
        g_hash[node.nodeID] = addr;
        g_nodes[addr] = node;
    }
    if (working && insert) {
        uint32_t index = s_heapInsertBase + heapIndex;
        g_heapInsertList[index].fValue = node.fValue;
        g_heapInsertList[index].addr = addr;
    }
}

template<int NB, int NT, int HEAP_CAPACITY>
__global__ void kHeapInsert(
    // open list
    heap_t g_openList[],
    int g_heapSize[],
    int *g_heapBeginIndex,

    heap_t g_heapInsertList[],
    int *g_heapInsertSize,

    // cleanup variable
    int *sortListSize,
    int *sortListSize2,
    uint32_t *optimalDistance,
    int *optimalNodesSize
)
{
    int gid = GLOBAL_ID;

    int heapInsertSize = *g_heapInsertSize;
    int heapIndex = *g_heapBeginIndex + gid;
    if (heapIndex >= NB*NT)
        heapIndex -= NB*NT;

    int heapSize = g_heapSize[heapIndex];
    heap_t *heap = g_openList + HEAP_CAPACITY * heapIndex - 1;

    for (int i = gid; i < heapInsertSize; i += NB*NT) {
        heap_t value = g_heapInsertList[i];
        int now = ++heapSize;

#ifdef KERNEL_LOG
        printf("\t\t\t[%d]: Push [%d] to heap %d\n",
               gid, value.addr, heapIndex);
#endif
        while (now > 1) {
            int next = now / 2;
            heap_t nextValue = heap[next];
            if (value < nextValue) {
                heap[now] = nextValue;
                now = next;
            } else
                break;
        }
        heap[now] = value;
    }

    g_heapSize[heapIndex] = heapSize;
    if (gid == 0) {
        *sortListSize = 0;
        *sortListSize2 = 0;
        *optimalDistance = UINT32_MAX;
        *optimalNodesSize = 0;
    }
}

__global__ void kFetchAnswer(
    node_t *g_nodes,

    uint32_t *lastAddr,

    uint32_t answerList[],
    int *g_answerSize
)
{
    int count = 0;
    int addr = *lastAddr;

    while (addr != UINT32_MAX) {
#ifdef KERNEL_LOG
        int x, y;
        idToXY(g_nodes[addr].nodeID, &x, &y);
        printf("\t\t\t Address: %d (%d, %d)\n", addr, x, y);
#endif
        answerList[count++] = g_nodes[addr].nodeID;
        addr = g_nodes[addr].prev;
    }

    *g_answerSize = count;
}

#endif /* end of include guard: __GPU_KERNEL_CUH_IUGANILK */
