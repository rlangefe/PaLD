import numpy as np
import cupy as cp
from tqdm import tqdm


ops = ('--use_fast_math',)

reach_set = cp.RawKernel(r'''
extern "C" __global__
void reach_set(const float* dx, const float* dy, const float &xy, const float &yx, float* Uxy, const int n) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        if ((dx[tid] <= xy) | (dy[tid] <= yx)) {
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
    'reduce_Uxy')

find_A3_x = cp.RawKernel(r'''
extern "C" __global__
void find_A3_x(const float* dx, const float* dy, const float* Uxy, float* A3_x, const float &sum_Uxy, const int n) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if ((tid < n) & (dx[tid]<=dy[tid])) {
        A3_x[tid]+=(Uxy[tid]/sum_Uxy);
    }
}
''','find_A3_x', options=ops)

def pald(D,block_size=32,dev=cp.cuda.Device(0)):
    #cp.memoize(True)
    D_gpu = cp.array(D, dtype=cp.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = cp.zeros((n,n), dtype=cp.float32)
    Uxy = cp.zeros((n,), dtype=cp.float32)
    
    for x in range(0,n):
        for y in range(0,n):
            if x != y:
                reach_set(((n//block_size)+1,), (min(block_size,n),), (D_gpu[x,:], D_gpu[y,:], D_gpu[x,y], D_gpu[y,x], Uxy, n))
                sum_Uxy = reduce_Uxy(Uxy)
                find_A3_x(((n//block_size)+1,), (min(block_size,n),), (D_gpu[x,:], D_gpu[y,:], Uxy, A3[x,:], sum_Uxy, n))
                #find_A3_x(((n//block_size)+1,), (min(block_size,n),), (D_gpu[x,:], D_gpu[y,:], Uxy, A3[x,:], reduce_Uxy(Uxy), n))
                
    #dev.synchronize()
    return A3/(n-1)
