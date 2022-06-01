import numpy as np

import tensorflow as tf


def pald(D):
    D_tf = tf.convert_to_tensor(D, dtype=tf.float32)
    bet=1
    h=0.5
    b=0
    n = D.shape[0]
    A3 = tf.zeros((n,n))
    for x in range(0,n):
        dx=D_tf[x,:] 
        for y in range(0,n):
            if x != y:
                dy=D_tf[y,:]
                Uxy=(dx<=bet*D_tf[x,y]) | (dy<=bet*D_tf[y,x]) # the reaching set
                wx = tf.cast(Uxy & (dx<dy),tf.float32)+h*tf.cast(Uxy & (dx==dy), tf.float32)
                #tmp = tf.zeros((1,n))
                #tmp[Uxy] = wx[Uxy]
                A3[x,:].assign(A3[x,:]+ 1/(tf.math.reduce_sum(tf.cast(Uxy, tf.float32)))*wx)
    
    return A3/(n-1)
