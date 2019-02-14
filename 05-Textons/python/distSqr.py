def distSqr(x,y):
    import numpy as np
    assert x.shape[0]==y.shape[0], 'x.shape[0]!=y.shape[0]' 

    [d,n] = x.shape
    [d,m] = y.shape

    z = np.matmul(x.transpose(),y)
    x2 = np.sum(x**2, axis=0)
    y2 = np.sum(y**2, axis=0)
    for i in range(m):
      z[:,i] = x2 + y2[i] - 2*z[:,i]

    return z
