import numpy as np
import cupy as cp
from tqdm import tqdm


ops = ('--use_fast_math', '--std=c++11')

reach_set = cp.RawKernel(r'''
extern "C" __global__
void reach_set(const float* d, const unsigned int x, const unsigned int y, const float &xy, const float &yx, float* Uxy, const int n) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        if ((d[x*n + tid] <= xy) | (d[y*n + tid] <= yx)) {
            Uxy[tid] = 1.0;
        } else {
            Uxy[tid] = 0.0;
        }
    }
}
''','reach_set', options=ops)

reduce_Uxy = cp.ReductionKernel(
    'float32 Uxy',
    'float32 sum_Uxy',
    'Uxy',
    'a + b',
    'sum_Uxy=a',
    '0',
    'reduce_Uxy', options=ops)

find_A3_x = cp.RawKernel(r'''
extern "C" __global__
void find_A3_x(const float* d, const unsigned int x, const unsigned int y, const float* Uxy, float* A3_x, const float &sum_Uxy, const int n) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if ((tid < n) & (d[x*n + tid]<=d[x*n + tid])) {
        A3_x[x*n + tid]+=(Uxy[tid]/sum_Uxy);
    }
}
''','find_A3_x', options=ops)

def pald(D,block_size=1024,dev=cp.cuda.Device(0)):
    D_gpu = D#cp.array(D, dtype=cp.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = cp.zeros((n,n), dtype=cp.float32)
    Uxy = cp.zeros((n,), dtype=cp.float32)
    
    for x in range(0,n):
        for y in range(0,n):
            if x != y:
                reach_set(((n//block_size)+1,), (min(block_size,n),), (D_gpu, x, y, D_gpu[x,y], D_gpu[y,x], Uxy, n))
                sum_Uxy = reduce_Uxy(Uxy)
                find_A3_x(((n//block_size)+1,), (min(block_size,n),), (D_gpu, x, y, Uxy, A3, sum_Uxy, n))
                
    dev.synchronize()
    return A3/(n-1)
