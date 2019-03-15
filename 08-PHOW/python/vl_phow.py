def imsmooth(img, sigma):
    from scipy.ndimage import gaussian_filter
    import numpy as np
    im = np.zeros(img.shape)
    for i in range(img.shape[-1]):
        im[:,:,i] = gaussian_filter(img[:,:,i], sigma)
    return im

def vl_phow(im, **kwargs):
# -------------------------------------------------------------------
# Parse the arguments
# -------------------------------------------------------------------
    from utils import NameSpace
    import math, warnings, os
    from skimage import color 
    from cyvlfeat.sift import dsift
    import numpy as np
    opts = NameSpace()
    opts.verbose = True 
    opts.fast = True 
    opts.sizes = [4, 6, 8, 10] 
    opts.step = 2 
    opts.color = 'gray' 
    opts.floatdescriptors = False 
    opts.magnif = 6 
    opts.windowsize = 1 #1.5 
    opts.contrastthreshold = 0.005 
    opts.update(**kwargs)

    dsiftOpts = {'step': opts.step, 'norm': True, 'window_size': opts.windowsize} 
    dsiftOpts['verbose'] = opts.verbose
    dsiftOpts['fast'] = opts.fast 
    dsiftOpts['float_descriptors'] = opts.floatdescriptors

# -------------------------------------------------------------------
# Extract the features
# -------------------------------------------------------------------


    # standarize the image
    imageSize = [im.shape[1], im.shape[0]] 
    if opts.color.lower() == 'gray':
        numChannels = 1 
        if len(im.shape) == 3: 
            im = np.expand_dims(color.rgb2gray(im), axis=-1)
    else:
        numChannels = 3 
        if len(im.shape) == 2:
            im = np.tile(np.expand_dims(im,-1), (1,1,numChannels))

        if opts.color.lower() == 'rgb':
            pass

        elif opts.color.lower() == 'opponent':
            #    Note that the mean differs from the standard definition of opponent
            #    space and is the regular intesity (for compatibility with
            #    the contrast thresholding).
            # 
            #    Note also that the mean is added pack to the other two
            #    components with a small multipliers for monochromatic
            #    regions.
            mu = 0.3*im[:,:,0] + 0.59*im[:,:,1] + 0.11*im[:,:,2] 
            alpha = 0.01 
            im = np.concatenate((mu, (im[:,:,0] - im[:,:,1])/math.sqrt(2) + alpha*mu, 
                            (im[:,:,0] + im[:,:,1] - 2*im[:,:,2])/math.sqrt(6) + alpha*mu), dim=-1)

        elif opts.color.lower() == 'hsv':
            im = color.rgb2hsv(im) 

        else:
            opts.color = 'hsv' 
            warnings.warn('Color space not recongized, defaulting to HSV color space.') 
        
    

    if opts.verbose:
        print('%s: color space: %s'%(os.path.basename(__file__), opts.color))
        print('%s: image size: %d x %d'%(os.path.basename(__file__), imageSize[0], imageSize[1]))
        print('%s: sizes: %s'%(os.path.basename(__file__), str(opts.sizes)))
    

    frames = []
    descrs = []
    for si in range(len(opts.sizes)):

        #    Recall from VL_DSIFT() that the first descriptor for scale SIZE has
        #    center located at XC = XMIN + 3/2 SIZE (the Y coordinate is
        #    similar). It is convenient to align the descriptors at different
        #    scales so that they have the same geometric centers. For the
        #    maximum size we pick XMIN = 1 and we get centers starting from
        #    XC = 1 + 3/2 MAX(OPTS.SIZES). For any other scale we pick XMIN so
        #    that XMIN + 3/2 SIZE = 1 + 3/2 MAX(OPTS.SIZES).
        # 
        #    In pracrice, the offset must be integer ('bounds'), so the
        #    alignment works properly only if all OPTS.SZES are even or odd.
        off = int(math.floor(3/2 * (max(opts.sizes) - opts.sizes[si])))

        #    smooth the image to the appropriate scale based on the size
        #    of the SIFT bins
        sigma = float(opts.sizes[si]) / opts.magnif 
        # ims = vl_imsmooth(im, sigma) # window of ceil(4 *sigma)
        try:ims = imsmooth(im, sigma)
        except: import ipdb; ipdb.set_trace()

        dsiftOpts['size'] = opts.sizes[si]
        dsiftOpts['bounds'] = [off, off, im.shape[0]-1, im.shape[1]-1]
        #    extract dense SIFT features from all channels
        f = []
        d = []
        for k in range(numChannels):
            _f, _d = dsift(ims[:,:,k], **dsiftOpts) 
            f.append(_f.T)
            d.append(_d.T)
        # import ipdb; ipdb.set_trace()
        
        #    remove low contrast descriptors
        #    note that for color descriptors the V component is
        #    thresholded
        if opts.color.lower() in ['gray', 'opponent']:
            contrast = f[0][-1] 
        elif opts.color.lower() == 'rgb':
            contrast = np.mean((f[0][-1], f[1][-1], f[2][-1]), axis=0) 
        else: #    hsv
            contrast = f[2][-1] 
        
        for k in range(numChannels):
            thresh = np.where(contrast < opts.contrastthreshold)[0]
            for i in thresh:
                d[k][:,i] = 0 
        
        #    save only x,y, and the scale
        frames.append(np.concatenate([f[0][0:-1],  opts.sizes[si] * np.ones((1, f[0].shape[1]))], axis=0))
        descrs.append(np.concatenate(d, axis=0))
    
    frames = np.concatenate(frames, axis=1)
    descrs = np.concatenate(descrs, axis=1)

    return frames, descrs
