import numpy as np
import cupy as cp
from tqdm import tqdm

reach_set = cp.RawKernel(r'''
extern "C" __global__
void reach_set(const float* d_mat, float* Uxy, const unsigned int n, const unsigned int x) {
    unsigned int tid_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tid_j = blockDim.y * blockIdx.y + threadIdx.y;
    if ((tid_i < n) & (tid_j < n)) {
        if (((d_mat[x*n + tid_i] <= d_mat[x*n + tid_j]) | (d_mat[tid_j*n + tid_i] <= d_mat[tid_j*n + x])) & (tid_j != x)) {
            Uxy[tid_j*n + tid_i] = 1.0f;
        } else {
            Uxy[tid_j*n + tid_i] = 0.0f;
        }        
    }
}
''','reach_set')

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
void find_A3_x(const float* d_mat, const float* Uxy, float* A3_x, const float *sum_Uxy, const unsigned int n, const unsigned int x) {
    unsigned int tid_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tid_j = blockDim.y * blockIdx.y + threadIdx.y;
    if ((tid_i < n) & (tid_j < n) & (tid_j != x) & (d_mat[x*n + tid_i]<d_mat[tid_j*n + tid_i])) {
        atomicAdd(&A3_x[tid_i], (Uxy[tid_j*n + tid_i]/sum_Uxy[tid_j]));
    }
}
''','find_A3_x')

def pald(D,block_size=32,dev=cp.cuda.Device(0)):
    cp.memoize(True)
    D_gpu = cp.array(D, dtype=cp.float32)
    n = D.shape[0]
    A3 = cp.zeros((n,n), dtype=cp.float32) 
    Uxy = cp.zeros((n,n), dtype=cp.float32)
    
    for x in range(0,n):
        reach_set(((n//block_size)+1,(n//block_size)+1), (min(block_size,n),min(block_size,n)), (D_gpu, Uxy, n, x))
        sum_Uxy = cp.sum(Uxy, axis=1)
        find_A3_x(((n//block_size)+1,(n//block_size)+1), (min(block_size,n),min(block_size,n)), (D_gpu, Uxy, A3[x,:], sum_Uxy, n, x))
        dev.synchronize()

    dev.synchronize()
    return A3/(n-1)

