import numpy as np
import cupy as cp
from tqdm import tqdm

reach_set = cp.RawKernel(r'''
#include <stdio.h>
extern "C" __global__
void reach_set(const float* dx, const float* dy, const float* xy, const float* yx, float* Uxy, const int n) {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((tidx < n) & (tidy < n)) {
        if ((tidx != tidy) & (dx[tidx] <= xy[tidy]) | (dy[n*tidx + tidy] <= yx[tidx])) {
            Uxy[n*tidx + tidy] = 1.0;
        } else {
            Uxy[n*tidx + tidy] = 0.0;
        }
    }
}
''','reach_set', jitify=True)

reduce_Uxy = cp.ReductionKernel(
    'float32 Uxy',
    'float32 sum_Uxy',
    'Uxy',
    'a + b',
    'sum_Uxy=a',
    '0',
    'reduce_Uxy'
)

find_A3_x = cp.RawKernel(r'''
extern "C" __global__
void find_A3_x(const float* dx, const float* dy, const float* Uxy, float* A3_x, const float* sum_Uxy, const int n) {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((tidx < n) & (tidy < n) & (tidx != tidy) & (dx[tidx]<=dy[n*tidx + tidy])) {
        atomicAdd(&A3_x[tidx],(Uxy[n*tidx + tidy]/sum_Uxy[tidx]));
    }
}
''','find_A3_x')

def pald(D,block_size=1024,stream_count=8,dev=cp.cuda.Device(0)):
    D_gpu = cp.array(D, dtype=cp.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = cp.zeros((n,n), dtype=cp.float32)
    Uxy = cp.zeros((n,n), dtype=cp.float32)
    dev.synchronize()

    for x in tqdm(range(0,n)):
        reach_set(((n//block_size)+1,(n//block_size)+1), (min(block_size,n),min(block_size,n)), (D_gpu[x,:], D_gpu, D_gpu[x,:], D_gpu[:,x], Uxy, n))
        dev.synchronize()
        sum_Uxy = reduce_Uxy(Uxy, axis=0)
        find_A3_x(((n//block_size)+1,(n//block_size)+1), (min(block_size,n),min(block_size,n)), (D_gpu[x,:], D_gpu, Uxy, A3[x,:], sum_Uxy, n))
    
    dev.synchronize()
    return A3/(n-1)
