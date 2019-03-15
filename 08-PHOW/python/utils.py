# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def Download_data(conf):
    import wget, os
    if not os.path.isdir(conf.calDir) or (not os.path.isdir(os.path.join(conf.calDir, 'airplanes')) and
            not os.path.isdir(os.path.join(conf.calDir, '101_ObjectCategories', 'airplanes'))):
        if not conf.autoDownloadData:
            raise TypeError('Caltech-101 data not found.\nSet conf.autoDownloadData=true to download the required data.') 
        if not os.path.isdir(conf.calDir):
            os.makedirs(conf.calDir) 
        calUrl = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
        print('Downloading Caltech-101 data to {}. This will take a while.'.format(conf.calDir))
        wget.download(calUrl)
        untar('101_ObjectCategories.tar.gz', conf.calDir) 
        os.remove('101_ObjectCategories.tar.gz')

    if not os.path.isdir(os.path.join(conf.calDir, 'airplanes')):
        conf.calDir = os.path.join(conf.calDir, '101_ObjectCategories') 


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
class NameSpace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        for key,value in kwargs.iteritems():
            setattr(self, key, value)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def untar(fname, where):
    import tarfile
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall(path=where)
        tar.close()
        print "Extracted in Current Directory"
    else:
        print "Not a tar.gz file: '%s'."%fname


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def standarizeImage(im, resize=None):
    import numpy as np
    im = im.astype(np.float32)
    if im.shape[0] > 480:
        import imutils
        im = imutils.resize(im, height=480)
    if resize is not None:
        import cv2
        im = cv2.resize(im,(resize, resize))
        # im = cv2.resize(im,(64, 64))
    return im


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def getImageDescriptor(model, im):
    import numpy as np
    from vl_phow import vl_phow
    im = standarizeImage(im, resize=200) 
    width = im.shape[1]
    height = im.shape[0] 
    numWords = model.vocab.shape[1]
    model.phowOpts['verbose'] = False

    # get PHOW features
    frames, descrs = vl_phow(im, **model.phowOpts) 

    # quantize local descriptors into visual words
    if model.quantizer == 'kdtree':
        binsa = model.kdtree.query(descrs.T.astype(np.float32))[1]

    hists = []
    for i in range(len(model.numSpatialX)):
        binsx = np.digitize(frames[1], np.linspace(0, width, model.numSpatialX[i]+1)) - 1
        binsy = np.digitize(frames[0], np.linspace(0, height, model.numSpatialY[i]+1)) - 1

        # combined quantization
        bins = np.ravel_multi_index((binsy, binsx, binsa), [model.numSpatialY[i], model.numSpatialX[i], numWords])
        hist = np.histogram(bins,len(bins))[0].astype(np.float32)
        hists.append(hist / np.sum(hist))
    hist = np.concatenate(hists, axis=0) 
    hist = hist / np.sum(hist) 
    return hist


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def classify(model, im):
    hist = getImageDescriptor(model, im) 
    psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) 
    scores = model.w.T * psix + model.b.T 
    score, best = max(scores) 
    className = model.classes[best]
    return className, score


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def save(path, param):
    import numpy as np
    np.save(path, param)


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def load(path):
    import numpy as np
    return np.load(path)
