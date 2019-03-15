#!/home/afromero/anaconda3/envs/python2/bin/ipython

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def random_subset(img, N):
    import random
    _img = list(img)
    random.seed(conf.randSeed)
    random.shuffle(_img)
    return _img[:N]


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
if __name__ == '__main__':
    import os, glob, argparse, imageio, tqdm
    import numpy as np
    from utils import Download_data, NameSpace, untar, load, save
    from utils import standarizeImage, getImageDescriptor, classify
    from vl_phow import vl_phow
    from cyvlfeat.kmeans import kmeans

    parser = argparse.ArgumentParser()
    parser.add_argument('--calDir', type=str, default='data/caltech-101')
    parser.add_argument('--dataDir', type=str, default='data/')
    parser.add_argument('--color', type=str, default='rgb', choices=['gray', 'rgb', 'hsv'])
    parser.add_argument('--numTrain', type=int, default=15)
    parser.add_argument('--numTest', type=int, default=15)
    parser.add_argument('--numClasses', type=int, default=101)
    parser.add_argument('--numWords', type=int, default=600)
    parser.add_argument('--resize', type=int, default=256)

    parser.add_argument('--numSpatialX', type=list, default=[2,4])
    parser.add_argument('--numSpatialY', type=list, default=[2,4])

    parser.add_argument('--quantizer', type=str, default='kdtree')

    parser.add_argument('--prefix', type=str, default='baseline')
    parser.add_argument('--randSeed', type=int, default=1)

    parser.add_argument('--autoDownloadData', action='store_true', default=True)
    parser.add_argument('--clobber', action='store_true', default=False)
    parser.add_argument('--tinyProblem', action='store_true', default=True)
    conf = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=int, default=10)
    parser.add_argument('--solver', type=str, default='sdca', choices=['sdca', 'sgd', 'liblinear'])
    parser.add_argument('--biasMultiplier', type=int, default=1)
    conf.svm = parser.parse_args()

    if conf.tinyProblem:
        conf.prefix = 'tiny' 
        conf.numClasses = 5 
        conf.numSpatialX = [2] 
        conf.numSpatialY = [2] 
        conf.numWords = 300 
        conf.phowOpts = {'verbose': True, 'sizes': [7], 'step': 5}
    else:
        conf.phowOpts = {'step': 3} 
    conf.phowOpts['color'] = conf.color


    conf.vocabPath = os.path.join(conf.dataDir, conf.prefix+'-vocab.npy') 
    conf.histPath = os.path.join(conf.dataDir, conf.prefix+'-hists.npy') 
    conf.modelPath = os.path.join(conf.dataDir, conf.prefix+'-model.npy') 
    conf.resultPath = os.path.join(conf.dataDir, conf.prefix+'-result') 

    # randn('state',conf.randSeed) 
    # rand('state',conf.randSeed) 
    # vl_twister('state',conf.randSeed) 

    # --------------------------------------------------------------------
    #    Download Caltech-101 data
    # --------------------------------------------------------------------
    Download_data(conf)

    # --------------------------------------------------------------------
    #                                 Setup data
    # --------------------------------------------------------------------
    classes = sorted(os.listdir(conf.calDir))
    classes = classes[:conf.numClasses]
    images = [] 
    imageClass = [] 
    for ci in range(len(classes)):
        ims = glob.glob(os.path.join(conf.calDir, classes[ci], '*.jpg'))
        ims = random_subset(ims, conf.numTrain + conf.numTest) 
        images.extend(ims)
        imageClass.extend([ci]*len(ims))

    selTrain = np.where(np.arange(len(images))%(conf.numTrain + conf.numTest)<conf.numTrain)[0].tolist()
    selTest = np.where(np.arange(len(images))%(conf.numTrain + conf.numTest)>=conf.numTrain)[0].tolist()

    model = NameSpace()
    model.classes = classes 
    model.phowOpts = conf.phowOpts 
    model.numSpatialX = conf.numSpatialX 
    model.numSpatialY = conf.numSpatialY 
    model.quantizer = conf.quantizer 
    model.vocab = [] 
    model.w = [] 
    model.b = [] 
    model.classify = classify 

    # --------------------------------------------------------------------
    #                     Train vocabulary
    # --------------------------------------------------------------------

    if not os.path.isfile(conf.vocabPath):

        # Get some PHOW descriptors to train the dictionary
        selTrainFeats = random_subset(selTrain, 30) 
        descrs = []
        for ii in range(len(selTrainFeats)):
            im = imageio.imread(images[selTrainFeats[ii]])
            im = standarizeImage(im)
            descrs.append(vl_phow(im, **model.phowOpts)[1])

        descrs = np.concatenate(descrs, axis=1).T
        descrs = random_subset(descrs.tolist(), int(40e4)) 
        descrs = np.array(descrs).T.astype(np.float32)

        # Quantize the descriptors to get the visual words
        vocab = kmeans(descrs.T.copy(order='C'), conf.numWords, verbose=True, algorithm='ELKAN', max_num_iterations=100) 
        # Required .copy(order='C') because of cython
        vocab = vocab.T
        save(conf.vocabPath, vocab) 
    else:
        vocab = load(conf.vocabPath) 

    model.vocab = vocab 

    if model.quantizer == 'kdtree':
        from scipy.spatial import KDTree
        model.kdtree = KDTree(vocab.T) 

    # --------------------------------------------------------------------
    # Compute spatial histograms
    # --------------------------------------------------------------------

    if not os.path.isfile(conf.histPath):
        hists = []
        for ii in tqdm.tqdm(range(len(images)), desc='Computing spatial histograms'): 
            im = imageio.imread(images[ii])
            hists.append(getImageDescriptor(model, im))

        hists = np.concatenate(hists, axis=1) 
        save(conf.histPath, hists) 
    else:
        hists = load(conf.histPath) 
     

    # # --------------------------------------------------------------------
    # #                Compute feature map
    # # --------------------------------------------------------------------

    # psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) 

    # # --------------------------------------------------------------------
    # #                                    Train SVM
    # # --------------------------------------------------------------------

    # if not os.path.isdir(conf.modelPath) || conf.clobber
    #     if conf.svm.solver in ['sgd', 'sdca']:
    #             _lambda = 1 / (conf.svm.C *    length(selTrain)) 
    #             w = [] 
    #             for ci in range(len(classes)):
    #                 perm = randperm(length(selTrain)) 
    #                 fprintf('Training model for class #s\n', classes{ci}) 
    #                 y = 2 * (imageClass(selTrain) == ci) - 1 
    #                 [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), _lambda, ...
    #                     'Solver', conf.svm.solver, ...
    #                     'MaxNumIterations', 50/_lambda, ...
    #                     'BiasMultiplier', conf.svm.biasMultiplier, ...
    #                     'Epsilon', 1e-3)

    #     elif  conf.svm.solver == 'liblinear':
    #         svm = train(imageClass(selTrain).T,
    #                                 sparse(double(psix(:,selTrain))),
    #                                 ' -s 3 -B {} -c {}'.format(
    #                                                 conf.svm.biasMultiplier, conf.svm.C),
    #                                 'col') 
    #         w = svm.w(:,1:-1).T
    #         b = svm.w(:,-1).T

    #     model.b = conf.svm.biasMultiplier * b 
    #     model.w = w 

    #     save(conf.modelPath, 'model') 
    # else
    #     load(conf.modelPath) 
    #  

    # # --------------------------------------------------------------------
    # #            Test SVM and evaluate
    # # --------------------------------------------------------------------

    # # Estimate the class of the test images
    # scores = model.w' * psix + model.b' * ones(1,size(psix,2)) 
    # [drop, imageEstClass] = max(scores, [], 1) 

    # # Compute the confusion matrix
    # idx = sub2ind([length(classes), length(classes)], ...
    #                             imageClass(selTest), imageEstClass(selTest)) 
    # confus = zeros(length(classes)) 
    # confus = vl_binsum(confus, ones(size(idx)), idx) 

    # # Plots
    # figure(1)  clf
    # subplot(1,2,1) 
    # imagesc(scores(:,[selTrain selTest]))  title('Scores') 
    # set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) 
    # subplot(1,2,2) 
    # imagesc(confus) 
    # title(sprintf('Confusion matrix (#.2f ## accuracy)', ...
    #                             100 * mean(diag(confus)/conf.numTest) )) 
    # print('-depsc2', [conf.resultPath '.ps']) 
    # save([conf.resultPath '.mat'], 'confus', 'conf') 
