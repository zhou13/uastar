#include <cstdio>
#include <iostream>
#include <moderngpu.cuh>

template<int NT, int VT, typename Key>
__global__ void BlockSortKernel(Key *d_in, Key *d_out)
{
    int tid = threadIdx.x;
    Key key[VT];

    __shared__ Key shared[NT*VT+1];

    mgpu::DeviceGlobalToThread<NT, VT>(NT*VT, d_in, tid, key);
    mgpu::CTAMergesortKeys<NT, VT, false>(key, shared, VT*NT, tid, mgpu::less<int>());
    mgpu::DeviceSharedToGlobal<NT, VT>(NT*VT, shared, tid, d_out);
}

int main(int argc, char *argv[])
{
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(argc, argv, true);

    const int vt = 2;
    MGPU_MEM(int32_t) d_in  = context->GenRandom<int32_t>(192 * vt, 1, 1000);
    MGPU_MEM(int32_t) d_out = context->Malloc<int32_t>(192 * vt);

    BlockSortKernel<192, vt, int32_t><<<1, 192>>>(*d_in, *d_out);

    // context->Start();
    // double elapsed = context->Split();

    // printf("Time elapsed: %.2f\n", elapsed);

    // puts("Input array: ");
    // mgpu::PrintArray(*d_in, "%6d", 12);
    // puts("Output array: ");
    // mgpu::PrintArray(*d_out, "%6d", 12);

    return 0;
}
