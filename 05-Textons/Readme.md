# Textons 

In this lab you will use a strategy to represent images using Textons. Then you will train, and evaluate a classifier based on the texton representation. 

Try to develop quality code so that you can reuse it in the following labs.

**The code for this lab takes a VERY LONG time to execute, plan ahead a use wisely the course servers, you are not the only one there!!!**

## Database

The database for this lab is the famous [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html):

<html>
<table>
    <tr>
        <td class="cifar-class-name">airplane</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">automobile</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">bird</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">cat</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">deer</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">dog</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">frog</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">horse</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">ship</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">truck</td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck1.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck2.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck3.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck4.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck5.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck6.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck7.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck8.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck9.png" class="cifar-sample" /></td>
        <td><img src="https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck10.png" class="cifar-sample" /></td>
    </tr>
</table>
</html>

Download the CIFAR-10 python version and uncompress it (it will create a folder  `cifar-10-batches-py`, with the binary files inside).  Please refer to `cifar10.py` for properly loading the data into python. 

​    
## Image Representation

The *lib* folder contains functions that can be used to represent images as textons.

Pay special attention to the following functions (you will see them again in the example), try to figure out what they do, what their inputs and outputs are.

    -   fbCreate
    -   fbRun
    -   computeTextons
    -   assignTextons

The following script will give you some ideas on how to create a texton dictionary from 2 sample images, then use it  to compare another 2 images:

**Matlab**

```Matlab
addpath('matlab')

%%Create a filter bank with deafult params
[fb] = fbCreate;

%%Load sample images from disk
imBase1=double(rgb2gray(imread('img/moto1.jpg')))/255;
imBase2=double(rgb2gray(imread('img/perro1.jpg')))/255;

%Set number of clusters
k = 16*8;

%Apply filterbank to sample image
filterResponses=fbRun(fb,horzcat(imBase1,imBase2))

%Computer textons from filter
[map,textons] = computeTextons(filterResponses,k);

%Load more images
imTest1=double(rgb2gray(imread('img/moto2.jpg')))/255;
imTest2=double(rgb2gray(imread('img/perro2.jpg')))/255;

%Calculate texton representation with current texton dictionary
tmapBase1 = assignTextons(fbRun(fb,imBase1),textons');
tmapBase2 = assignTextons(fbRun(fb,imBase2),textons');
tmapTest1 = assignTextons(fbRun(fb,imTest1),textons');
tmapTest2 = assignTextons(fbRun(fb,imTest2),textons');

%Check the euclidean distances between the histograms and convince yourself that the images of the bikes are closer because they have similar texture pattern
%Can you tell why we need to create a histogram before measuring the distance?
D = norm(histc(tmapBase1(:),1:k)/numel(tmapBase1) - histc(tmapTest1(:),1:k)/numel(tmapTest1))
D = norm(histc(tmapBase1(:),1:k)/numel(tmapBase1) - histc(tmapTest2(:),1:k)/numel(tmapTest2))

D = norm(histc(tmapBase2(:),1:k)/numel(tmapBase2) - histc(tmapTest1(:),1:k)/numel(tmapTest1))
D = norm(histc(tmapBase2(:),1:k)/numel(tmapBase2)  - histc(tmapTest2(:),1:k)/numel(tmapTest2))
```

**Python**

```python
import sys
sys.path.append('python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

#Load sample images from disk
from skimage import color
from skimage import io
from skimage.transform import resize

imBase1=color.rgb2gray(resize(io.imread('img/moto1.jpg'), (32, 32)))
imBase2=color.rgb2gray(resize(io.imread('img/perro1.jpg'), (32, 32)))

#Set number of clusters
k = 16*8

#Apply filterbank to sample image
from fbRun import fbRun
import numpy as np
filterResponses = fbRun(fb,np.hstack((imBase1,imBase2)))

#Computer textons from filter
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

#Load more images
imTest1=color.rgb2gray(resize(io.imread('img/moto2.jpg'), (32, 32)))
imTest2=color.rgb2gray(resize(io.imread('img/perro2.jpg'), (32, 32)))

#Calculate texton representation with current texton dictionary
from assignTextons import assignTextons
tmapBase1 = assignTextons(fbRun(fb,imBase1),textons.transpose())
tmapBase2 = assignTextons(fbRun(fb,imBase2),textons.transpose())
tmapTest1 = assignTextons(fbRun(fb,imTest1),textons.transpose())
tmapTest2 = assignTextons(fbRun(fb,imTest2),textons.transpose())

#Check the euclidean distances between the histograms and convince yourself that the images of the bikes are closer because they have similar texture pattern

# --> Can you tell why do we need to create a histogram before measuring the distance? <---

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
	 histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
	 histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)

D = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
	 histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
	 histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)

```

## Classification

After the images are represented using a texton dictionary, train and evaluate a classifier using the provided database.

-   **Nearest neighbour:** Use intersection of histograms or Chi-Square metrics.

	Matlab: 
	- [KNN Clasifiers](https://www.mathworks.com/help/stats/classification-using-nearest-neighbors.html#btap7k2)
	- [distance metrics] (https://www.mathworks.com/help/stats/classification-using-nearest-neighbors.html).

	Python
	- [scikit - NN](http://scikit-learn.org/stable/modules/neighbors.html).
	- [scikit - KNN](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

- **Random forest:** 

  Matlab:
  - Use the matlab [tree bagger](http://www.mathworks.com/help/stats/treebagger.html) function. See an example at [kawahara.ca](http://kawahara.ca/matlab-treebagger-example/)

  Python:

  - [scikit - RF](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

Train both classifiers **only** with images from the *train* directory and then test them with images **ONLY** from the *test* directory. Provide the confusion matrix for training and test datasets. 

## Your Turn

The report for this laboratory must include:

-   Small (one or two paragraphs) description of the database.
-   Overall description of the method and filters used for representing the images
    -   How can we classify an image using textons? (don't be overly detailed on this, just one or two paragraphs)
    -   What does the texton representation of an image tell us?
    -   How did you create the dictionary?
    -   How many textons are you using? Why?
    -   Can you tell if some filters are more discriminative than others? Why would this happen?
-   Description of the classifiers, hyperparameters and distance metrics
    -   What hyperparameters can you find in the classifiers? How can you choose their values?
    -   Did you apply any adjustments or preprocessing to the data? why?
-   Results
    - Provide the confusion matrix for the training and test sets, what is it telling you?. 
    - Do you have another metric to measure the performance of your method? why do you need it?
-   Discussion of the results
    -   Which classifier works best? Any idea why?
    -   How much time does it takes to create the texton dictionary? Why is it so slow?
    -   How much time does it takes to train and apply both kinds of classifiers?
    -   What categories cause the most confusion? Could you give some insight on why this happens?
    -   What are the limitations of the method? (CPU and RAM constraints are well known limitations, go beyond this!!)
    -   How could your method be improved?

### Some advice on creating your texton dictionary
Creating a large texton dictionary takes A LOT of time and uses A LOT of memory. Creating a dictionary with the full set of training images **is not possible, on the course server**. Unless you can find a larger machine on your own, you will have to subsample the training database. Your subsample strategy is critical for the success in this laboratory, be sure to report and discuss it. Do not forget that textures are **local patterns** that repeat over the whole images.

### Report

The report should have max 5 pages written **in English**, if necessary use any additional pages **for references and images only**.
Use the standard CVPR sections: abstract, introduction (be concise), method, results, conclusions and references. There is no need for an 'State of the art' section. 

Upload the code you used into a directory named 'code', by extraordinary results (either good or bad), I will check it. There **MUST** be at least three different Python (.py) / Matlab (.m) files:

1. run.py / run.m: Runs the entire system: training and the confusion matrix over the test set.
2. test.py / test.m: Load the model previously trained and depicts the confusion matrix over the test set.
3. demo.py / demo.m: Load the model and displays several test images with their corresponding texton representation and classification. Different images always. 

## Deadline 
**February 28 11:59 pm.**

----

## Tip for ipython debugging from terminal
As I told you, you can debug in Python as you usually does it in Matlab (those funny red points that you click on). For Python is a little bit different as you do not have an interface (unless you have Spyder, but, let's assume you do not). For this purpose you can use `ipython` (you might have noticed that I love it) and a module called `ipdb`.

Just `import ipdb` and call `ipdb.set_trace()` wherever you want the code to make a pause and interact with it. That's it. enjoy it. **It really changed my life**. 

----

## DISCLAIMER
I have created the `Python scripts` from the matlab ones. Therefore, they are in *beta* mode. Please, do not lose your mind if you find any bug or error. If so, kindly report this bug to me (create an Issue in the repo), I'll be happy to fix it. 

