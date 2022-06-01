import numpy as np
import cupy as cp
from tqdm import tqdm
from cupyx import optimizing

def pald(D,dev=cp.cuda.Device(0)):
    D_gpu = cp.array(D, dtype=cp.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = cp.zeros((n,n))
    
    for x in range(0,n):
        for y in range(0,n):
            if x != y:
                Uxy=(D_gpu[x,:]<=bet*D_gpu[x,y]) | (D_gpu[y,:]<=bet*D_gpu[y,x])
                wx = (Uxy & (D_gpu[x,:]<D_gpu[y,:])).astype(cp.float32)     
                A3[x,:] = A3[x,:] + 1/(cp.sum(Uxy.astype(cp.float32)))*wx

    return A3/(n-1)
