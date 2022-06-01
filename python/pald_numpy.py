import numpy as np
#from tqdm import tqdm

def pald(D):
    D_data = D.astype(np.float32)
    n = D.shape[0]
    A3 = np.zeros((n,n))
    
    for x in range(0,n):
        for y in range(0,n):
            if x != y:
                Uxy=(D_data[x,:]<=D_data[x,y]) | (D_data[y,:]<=D_data[y,x])
                wx = (Uxy & (D_data[x,:]<D_data[y,:])).astype(np.float32)
                A3[x,:] = A3[x,:] + 1/(np.sum(Uxy.astype(np.float32)))*wx
    
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