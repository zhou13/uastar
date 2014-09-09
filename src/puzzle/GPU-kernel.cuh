#ifndef __GPU_KERNEL_CUH_IUGANILK
#define __GPU_KERNEL_CUH_IUGANILK

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <moderngpu.cuh>

#include "utils.hpp"
#include "puzzle/storage.hpp"

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
    uint8_t fValue;
    uint32_t addr;
};
inline __host__ __device__ bool operator<(const heap_t &a, const heap_t &b) {
    return a.fValue < b.fValue;
}
inline __host__ __device__ bool operator>(const heap_t &a, const heap_t &b) {
    return a.fValue > b.fValue;
}


template<int N>
struct node_t {
    PuzzleStorage<N> ps;
    uint32_t prev;
    uint8_t fValue;
    uint8_t gValue;
};

template<int N>
struct sort_t {
    PuzzleStorage<N> ps;
    uint8_t gValue;
    __host__ __device__ bool operator<(const sort_t<N> &r) {
        if (ps != r.ps)
            return ps < r.ps;
        return gValue < r.gValue;
    }
};

__constant__ int2 d_mapTracked[36];
__constant__ PuzzleStorage<3> d_target3;
__constant__ PuzzleStorage<4> d_target4;
__constant__ PuzzleStorage<5> d_target5;

template<typename T>
inline __device__ void swap(T &a, T &b)
{
    T t;
    t = a;
    a = b;
    b = t;
}

template<int N>
inline __device__ float inrange(int x, int y)
{
    return 0 <= x && x < N && 0 <= y && y < N;
}

template<int N>
inline __device__ int computeHValue(
    uint8_t database[],
    uint8_t conf[N][N],
    uint8_t index[][N == 5 ? 6 : 8])
{
    const int DB_COUNT[6] = { 0, 0, 0, 1, 2, 4 };
    const int DB_OFFSET[6][4] = {
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 518918400, 0, 0 },
        { 0, 127512000, 127512000 * 2, 127512000 * 3 },
    };
    const int INDEX_COUNT[6][4] = {
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 8, 0, 0, 0 },
        { 8, 7, 0, 0 },
        { 6, 6, 6, 6 },
    };
    const int MULTIPLE[4][9] = {
        { 362880, 40320, 5040, 720, 120, 24, 6, 2, 1 },
        { 518918400, 32432400, 2162160, 154440, 11880, 990, 90, 9, 1 },
        { 57657600, 3603600, 240240, 17160, 1320, 110, 10, 1, 0 },
        { 127512000, 5100480, 212520, 9240, 420, 20, 1, 0, 0},
    };

#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int j = 0; j < N; ++j)
            if (conf[i][j]) {
                int2 p = d_mapTracked[conf[i][j]];
                index[p.x][p.y] = i*N + j;
            }

    int retn = 0;
#pragma unroll
    for (int k = 0; k < DB_COUNT[N]; ++k) {
        int oindex[10];
#pragma unroll
        for (int i = 0; i < INDEX_COUNT[N][k]; ++i)
            oindex[i] = index[k][i];

#pragma unroll
        for (int i = 0; i < INDEX_COUNT[N][k]; ++i)
#pragma unroll
            for (int j = i+1; j < INDEX_COUNT[N][k]; ++j)
                if (oindex[i] < oindex[j])
                    --index[k][j];

        int mid;
        if (N == 3)
            mid = 0;
        else if (N == 4 && k == 0)
            mid = 1;
        else if (N == 4 && k == 1)
            mid = 2;
        else if (N == 5)
            mid = 3;

        uint32_t code = 0;
#pragma unroll
        for (int i = 0; i < INDEX_COUNT[N][k]; ++i)
            code += index[k][i] * MULTIPLE[mid][i+1];

        retn += database[DB_OFFSET[N][k] + code];
    }

    return retn;
}

template<int N>
inline __device__ void getEmptyTile(uint8_t conf[N][N], int *x, int *y)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (conf[i][j] == 0) {
                *x = i;
                *y = j;
                return;
            }
}

template<int N>
inline cudaError_t initializeCUDAConstantMemory(
    int2 mapTracked[],
    const PuzzleStorage<3> &target3,
    const PuzzleStorage<4> &target4,
    const PuzzleStorage<5> &target5
)
{
    cudaError_t ret = cudaSuccess;
    ret = cudaMemcpyToSymbol(d_mapTracked, mapTracked, sizeof(int2) * N*N);
    ret = cudaMemcpyToSymbol(d_target3, &target3, sizeof(target3));
    ret = cudaMemcpyToSymbol(d_target4, &target4, sizeof(target4));
    ret = cudaMemcpyToSymbol(d_target5, &target5, sizeof(target5));
    return ret;
}

template<int N, int M>
__global__ void kInitialize(
    PuzzleStorage<N> ps,
    uint8_t g_database[],
    node_t<N> g_nodes[],
    uint32_t g_hash[],
    heap_t g_openList[],
    int g_heapSize[]
)
{
    __shared__ uint8_t s_index[N==5 ? 4 : (N==4 ? 2 : 1)][N==5 ? 6 : 8];
    uint8_t conf[N][N];

    ps.decompose(conf);

    node_t<N> node;
    node.fValue = computeHValue<N>(g_database, conf, s_index);
    node.gValue = 0;
    node.prev = UINT32_MAX;
    node.ps = ps;

    heap_t heap;
    heap.fValue = node.fValue;
    heap.addr = 0;

    g_nodes[0] = node;
    g_openList[0] = heap;
    g_hash[ps.hashValue() % M] = 0;
    g_heapSize[0] = 1;
}

template<int N> __device__ bool checkSolution(const PuzzleStorage<N> &ps) { return true; }
template<> __device__ bool checkSolution<3>(const PuzzleStorage<3> &ps)
{
    return ps == d_target3;
}
template<> __device__ bool checkSolution<4>(const PuzzleStorage<4> &ps)
{
    return ps == d_target4;
}
template<> __device__ bool checkSolution<5>(const PuzzleStorage<5> &ps)
{
    return ps == d_target5;
}


// NB: number of CUDA block
// NT: number of CUDA thread per CUDA block
// VT: value handled per thread
template<int N, int NB, int NT, int HEAP_CAPACITY, int M>
__global__ void kExtractExpand(
    uint8_t g_database[],

    // global nodes
    node_t<N> g_nodes[],
    int *g_nodeSize,

    // open list
    heap_t g_openList[],
    int g_heapSize[],

    uint32_t g_hash[],

    // solution
    uint32_t *g_optimalStep,
    heap_t g_optimalNodes[],
    int *g_optimalNodesSize,

    // heap insert list
    heap_t *g_heapInsertList,
    int *g_heapInsertSize,

    // cleanup
    int *g_heapBeginIndex
)
{
    __shared__ uint32_t s_optimalStep;
    __shared__ uint8_t s_index[NT][N==5 ? 4 : (N==4 ? 2 : 1)][N==5 ? 6 : 8];

    __shared__ int s_nodeInsertCount;
    __shared__ int s_nodeInsertBase;

    __shared__ int s_heapInsertCount;
    __shared__ int s_heapInsertBase;

    int gid = GLOBAL_ID;
    int tid = THREAD_ID;
    if (tid == 0) {
        s_optimalStep = UINT32_MAX;
        s_nodeInsertCount = 0;
        s_heapInsertCount = 0;
    }

    __syncthreads();

    heap_t *heap = g_openList + HEAP_CAPACITY * gid - 1;

    heap_t topNode;
    int heapSize = g_heapSize[gid];
    bool working = (heapSize != 0);

    node_t<N> node;
    uint8_t conf[N][N];

    if (working) {
        topNode = heap[1];
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
        g_heapSize[gid] = heapSize;

        atomicMin(&s_optimalStep, topNode.fValue);

        // check solution
        node = g_nodes[topNode.addr];
        if (checkSolution<N>(node.ps)) {
            int index = atomicAdd(g_optimalNodesSize, 1);
            g_optimalNodes[index] = topNode;
            working = false;
        }
    }

    const int DX[4] = { 1, -1,  0,  0 };
    const int DY[4] = { 0,  0,  1, -1 };

    node_t<N> nnode[4];
    uint32_t addr[4];
    uint32_t hashValue[4];
    bool found[4], insert[4];

    int nodeCount = 0, nodeIndex;
    int heapCount = 0, heapIndex;

    if (working) {
        node.ps.decompose(conf);

        int x, y;
        getEmptyTile<N>(conf, &x, &y);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            int nx = x + DX[k];
            int ny = y + DY[k];

            found[k] = false;
            insert[k] = true;

            if (inrange<N>(nx, ny)) {
                PuzzleStorage<N> nps = PuzzleStorage<N>(conf);
                hashValue[k] = nps.hashValue() % M;
                addr[k] = g_hash[hashValue[k]];

                nnode[k].ps = nps;
                nnode[k].gValue = node.gValue + 1;
                nnode[k].fValue = nnode[k].gValue +
                    computeHValue<N>(g_database, conf, s_index[tid]);
                nnode[k].prev = topNode.addr;

                if (addr[k] != UINT32_MAX) {
                    node_t<N> onode = g_nodes[addr[k]];
                    if (onode.ps == nps) {
                        found[k] = true;
                        if (nnode[k].gValue < onode.gValue)
                            g_nodes[addr[k]] = nnode[k];
                        else
                            insert[k] = false;
                    }
                }
            }
            if (!found[k])
                ++nodeCount;
            if (insert[k])
                ++heapCount;
        }
        nodeIndex = atomicAdd(&s_nodeInsertCount, nodeCount);
        heapIndex = atomicAdd(&s_heapInsertCount, heapCount);
    }

    __syncthreads();
    if (tid == 0) {
        s_nodeInsertBase = atomicAdd(g_nodeSize, s_nodeInsertCount);
        s_heapInsertBase = atomicAdd(g_heapInsertSize, s_heapInsertCount);
    }
    __syncthreads();

    if (working) {
        nodeCount = 0;
        heapCount = 0;
        for (int k = 0; k < 4; ++k) {
            if (!found[k]) {
                addr[k] = s_nodeInsertBase + nodeIndex + nodeCount++;
                g_nodes[addr[k]] = nnode[k];
                __threadfence();
                g_hash[hashValue[k]] = addr[k];
            }
            if (insert[k]) {
                heap_t heapItem;
                heapItem.fValue = nnode[k].fValue;
                heapItem.addr = addr[k];
                g_heapInsertList[
                    s_heapInsertBase + heapIndex + heapCount++] = heapItem;
            }
        }
    }

    if (tid == 0)
        atomicMin(g_optimalStep, s_optimalStep);
    if (gid == 0) {
        int newHeapBeginIndex = *g_heapBeginIndex + *g_heapInsertSize;

        *g_heapBeginIndex = newHeapBeginIndex % (NB*NT);
        *g_heapInsertSize = 0;
    }
}

template<int N, int NB, int NT, int HEAP_CAPACITY>
__global__ void kHeapInsert(
    // open list
    heap_t g_openList[],
    int g_heapSize[],
    int *g_heapBeginIndex,

    heap_t g_heapInsertList[],
    int *g_heapInsertSize,

    // cleanup variable
    uint32_t *optimalStep,
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
        *optimalStep = UINT32_MAX;
        *optimalNodesSize = 0;
    }
}

template<int N>
__global__ void kFetchAnswer(
    node_t<N> g_nodes[],

    uint32_t *lastAddr,

    uint32_t answerList[],
    int *g_answerSize
)
{
    int count = 0;
    int addr = *lastAddr;

    while (addr != UINT32_MAX) {
        addr = g_nodes[addr].prev;
    }

    *g_answerSize = count;
}

#endif /* end of include guard: __GPU_KERNEL_CUH_IUGANILK */
