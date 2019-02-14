import numpy as np

def isum(x,idx,nbins):
    acc = np.zeros((nbins,))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if idx[i,j]<1: continue
            if idx[i,j]>nbins: continue
            acc[idx[i,j]] = acc[idx[i,j]] + x[i,j]
    return acc

