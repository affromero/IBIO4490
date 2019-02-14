def computeTextons(fim,k):
    import numpy as np
    from sklearn.cluster import KMeans
    fim = np.array(fim)
    fim_shape = fim.shape
    fim = fim.reshape(fim_shape[0]*fim_shape[1], -1)

    kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(fim.transpose()) #Ensuring KMeans has the same parameters as the Matlab function

    map = kmeans.labels_
    map = map.reshape(fim_shape[2], fim_shape[3])

    textons = kmeans.cluster_centers_

    return map, textons
