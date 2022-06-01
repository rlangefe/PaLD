import numpy as np
import cupy as cp

def pald(D):
    D_gpu = cp.array(D, dtype=cp.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = cp.zeros((n,n))
    for x in range(0,n):
        dx=D_gpu[x,:]
        dy=D_gpu
        Uxy=(dx<=bet*D_gpu[x,:].T) | (dy<=bet*D_gpu[:,x]) # the reaching set
        wx = (Uxy & (dx<dy)).astype(cp.float32)+h*(Uxy & (dx==dy)).astype(cp.float32)
        A3[x,:] = A3[x,:] + cp.sum(1/(cp.sum(Uxy.astype(cp.float32), axis=1))*wx, axis=0)

        # y=x
        # dy=D_gpu[y,:]
        # Uxy=(dx<=bet*D_gpu[x,y]) | (dy<=bet*D_gpu[y,x]) # the reaching set
        # wx = (Uxy & (dx<dy)).astype(cp.float32)+h*(Uxy & (dx==dy)).astype(cp.float32)
        # A3[x,:] = A3[x,:] - 1/(cp.sum(Uxy.astype(cp.float32)))*wx
    
    return A3/(n-1)
