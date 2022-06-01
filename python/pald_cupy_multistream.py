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
    //if ((tid_i < n) & (tid_j < n) & (tid_j != x) & (dx[tid_i]<dy[tid_j*n + tid_i])) {
    if ((tid_i < n) & (tid_j < n) & (tid_j != x) & (d_mat[x*n + tid_i]<d_mat[tid_j*n + tid_i])) {
        atomicAdd(&A3_x[tid_i], (Uxy[tid_j*n + tid_i]/sum_Uxy[tid_j]));
    }
}
''','find_A3_x')

def pald(D,block_size=16,dev=cp.cuda.Device(0), streams=4):
    with dev as curr_dev:
        D_gpu = cp.array(D, dtype=cp.float32)
        bet=1
        h=0.5
        b=0
        n = D.shape[0]
        A3 = cp.zeros((n,n), dtype=cp.float32)
        
        Uxy_list = []
        sum_Uxy = []

        map_streams = []
        for i in range(streams):
            map_streams.append(cp.cuda.stream.Stream(non_blocking=False))
            Uxy_list.append(cp.zeros((n,n), dtype=cp.float32))
            sum_Uxy.append(0)

        for x in range(0,n):
            with map_streams[x%streams] as s:
                reach_set(((n//block_size)+1,(n//block_size)+1), (min(block_size,n),min(block_size,n)), (D_gpu, Uxy_list[x%streams], cp.int32(n), x))
                #sum_Uxy[x%streams] = reduce_Uxy(Uxy_list[x%streams], axis=1)
                sum_Uxy[x%streams] = cp.sum(Uxy_list[x%streams], axis=1)
                find_A3_x(((n//block_size)+1,(n//block_size)+1), (min(block_size,n),min(block_size,n)), (D_gpu, Uxy_list[x%streams], A3[x,:], sum_Uxy[x%streams], n, x))
                s.synchronize()
                sum_Uxy[x%streams] = 0
        
        for s in map_streams:
            s.synchronize()

        curr_dev.synchronize()
        return A3/(n-1)
