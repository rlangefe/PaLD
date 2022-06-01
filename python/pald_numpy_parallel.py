import numpy as np
import multiprocessing as mp
#from tqdm import tqdm

def pald(D):
    D_gpu = D.astype(np.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = np.zeros((n,n))
    #for x in tqdm(range(0,n)):

    def loop_function(x):
        A3 = np.zeros(n)
        for y in range(0,n):
            if x != y:
                Uxy=(D_gpu[x,:]<=bet*D_gpu[x,y]) | (D_gpu[y,:]<=bet*D_gpu[y,x])
                wx = (Uxy & (D_gpu[x,:]<D_gpu[y,:])).astype(np.float32)
                A3 = A3 + 1/(np.sum(Uxy.astype(np.float32)))*wx

    with mp.Pool(processes = 2) as p:
        results = p.map(loop_function, range(0,n))
        
    A3 = np.array(results)
    
    return A3/(n-1)

def pald_predict(D):
    D_gpu = D.astype(np.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = np.zeros((n,n))
    #for x in tqdm(range(0,n)):
    for x in range(0,n):
        for y in range(0,n):
            if x != y:
                Uxy=(D_gpu[x,:]<=bet*D_gpu[x,y]) | (D_gpu[y,:]<=bet*D_gpu[y,x])
                wx = (Uxy & (D_gpu[x,:]<D_gpu[y,:])).astype(np.float32)
                A3[x,:] = A3[x,:] + 1/(np.sum(Uxy.astype(np.float32)))*wx
    
    return A3/(n-1)