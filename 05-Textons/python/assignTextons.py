def assignTextons(fim, textons):
    import numpy as np
    from distSqr import distSqr
    fim = np.array(fim)
    fim_shape = fim.shape
    fim = fim.reshape(fim_shape[0]*fim_shape[1], -1)
            
    d2 = distSqr(np.array(fim), np.array(textons))
    # y = np.min(d2, axis=1)
    map = np.argmin(d2, axis=1)
    map = map.reshape(fim_shape[2], fim_shape[3])
    return map
