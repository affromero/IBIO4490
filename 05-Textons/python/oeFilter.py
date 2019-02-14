import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal
from isum import isum
def oeFilter(sigma, support=3, theta=0, deriv=0, hil=0, vis=0):

    if type(sigma)==list and len(sigma)==1:
        sigma = np.array(sigma*2)
    elif type(sigma)==int or type(sigma)==float:
        sigma = np.array([int(sigma), int(sigma)])        

    assert deriv>0 or deriv<2, 'deriv must be in [0,2]'

    #Calculate filter size; make sure it's odd.
    hsz = max([math.ceil(i) for i in (support*np.array(sigma))])
    sz = int(2*hsz + 1)

    #Sampling limits.
    maxsamples = 1000 #Max samples in each dimension.
    maxrate = 10 #Maximum sampling rate.
    frate = 10 #Over-sampling rate for function evaluation.

    #Cacluate sampling rate and number of samples.
    rate = float(min(maxrate,max(1,math.floor(maxsamples/sz))))
    samples = sz*rate

    #The 2D samping grid.
    r = math.floor(sz/2) + 0.5 * (1 - (1./rate))
    dom = np.linspace(-r,r,samples)
    [sx,sy] = np.meshgrid(dom,dom)

    #Bin membership for 2D grid points.
    mx = np.round(sx)
    my = np.round(sy)
    membership = (mx+hsz+1) + (my+hsz)*sz

    #Rotate the 2D sampling grid by theta.
    su = sx*math.sin(theta) + sy*math.cos(theta)
    sv = sx*math.cos(theta) - sy*math.sin(theta)

    #Evaluate the function separably on a finer grid.
    R = r*math.sqrt(2)*1.01                 #radius of domain, enlarged by >sqrt(2)
    fsamples = math.ceil(R*rate*frate)         #number of samples
    fsamples = fsamples + (fsamples+1)%2     #must be odd
    fdom = np.linspace(-R,R,int(fsamples))    #domain for function evaluation
    gap = 2*R/(fsamples-1)                     #distance between samples

    #The function is a Gaussian in the x direction...
    fx = np.exp(-fdom**2/(2*sigma[0]**2))
    # ... and a Gaussian derivative in the y direction...
    fy = np.exp(-fdom**2/(2*sigma[1]**2))
    if deriv == 1:
        fy = fy * (-fdom/sigma[1]**2)
    elif deriv==2:
        fy = fy * ((fdom**2)/(sigma[1]**2) - 1)


    #...with an optional Hilbert transform.
    if hil:
        fy = scipy.signal.hilbert(fy).imag

    #Evaluate the function with NN interpolation.
    xi = np.round(su/gap) + np.floor(fsamples/2) + 1
    yi = np.round(sv/gap) + np.floor(fsamples/2) + 1
    f = fx[xi.astype(np.uint16)] * fy[yi.astype(np.uint16)]

    #Accumulate the samples into each bin.
    f = isum(f,membership.astype(np.uint16)-1,sz*sz)
    f = f.reshape(sz,sz)

    #zero mean
    if deriv>0:
        f = f - np.mean(f)

    #unit L1-norm
    sumf = np.sum(np.abs(f))
    if sumf>0:
        f = f / sumf

    if vis:
        plt.imshow(f.transpose())
        plt.show()

    return f.transpose()
