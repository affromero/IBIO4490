import math
import imageio
import numpy as np
import scipy.signal
from padReflect import padReflect

def fbRun(fb,im):
    #find the max filter size
    maxsz = fb[0][0].shape
    for filter in fb:
        for scale in filter:
            maxsz = max(maxsz, scale.shape)

    maxsz = maxsz[0]
    #pad the image 
    r = int(math.floor(maxsz/2))
    impad = padReflect(im,r)

    #run the filterbank on the padded image, and crop the result back
    #to the original image size
    fim = np.zeros(np.array(fb).shape).tolist()
    for i in range(np.array(fb).shape[0]):
        for j in range(np.array(fb).shape[1]):
            if fb[i][j].shape[0]<50:
                fim[i][j] = scipy.signal.convolve2d(impad, fb[i][j], 'same')
            else:
                fim[i][j] = fftconvolve(impad,fb[i][j])
            fim[i][j] = fim[i][j][r:-r,r:-r]
    return fim
