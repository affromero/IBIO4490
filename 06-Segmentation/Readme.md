# Segmentation Lab

In this lab you will code several well known segemetation strategies and then apply them on a small set of images with grundtruth annotations. 

Try to develop quality code, you will be using it for the next Lab.

## Database


The data for this lab can be downloaded from the course server using http:

- http://157.253.196.67/BSDS_small.zip

It is a small, randomly selected, subset of the BSDS500 Segmentation database (annotations included). 

## Creating a  basic segmentation method

Implement your own segmentation method using what you have learned in class. It should be a matlab/python function with the following signature:

Matlab

```matlab
function segmentation = segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters)
```

Python


```python
def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    ...
    return segmentation
```

Where

- colorSpace : 'rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy' or 'hsv+xy'
- clusteringMethod = 'kmeans', 'gmm', 'hierarchical' or 'watershed'.
- numberOfClusters positive integer (larger than 2)

If the input of the function is **MxNxC**, the output should be **MxN**, where each pixel has a cluster label. In other words, pixels that belong to the same cluster have the same positive integer. Be aware of the colorization map, which shall be constant in order to visually compare across the _clusteringMethod_. 


### Notes
- You don't have to implement all the code from scratch, use the built in functions of your language.
- If you run into memory problems try sampling down the images. 
- *xy* stands for the spatial _x_ and _y_ coordinates. It does not matter which pixel of the image you label as  0,0 just be consistent, and do not forget to report it.
- You may need to normalize some channels to make them comparable, or to make some of them more/less important in the clustering process.


## Test your function

Before testing your function make sure the segmentation results are somewhat acceptable. To visualize the output of your function use something like:

```python
plt.imshow(segm, cmap=plt.get_cmap('tab20b')) #or another colormap that you like https://matplotlib.org/examples/color/colormaps_reference.html
plt.show()
```

You should see an image similar to the following, where each color represents a different cluster.

![Example of segmentation](imgs/segmented.png)

Try different parameters of your function, and inspect them visually before actually testing them.


### Reading ground truth data

The file 'BSDS_small.zip' contains the segmentation ground truth data. It has the same name as the image file and is saved as matlab file (.mat). The mat file contains several manual segmentation created by human annotators.

For example, to look at the ground truth for image ``train/22090`` we can use the following code for Python:

```python
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio

img = 'BSDS_small/train/22090.jpg'
plt.imshow(imageio.imread(img))
plt.show()

# Load .mat
gt=sio.loadmat(img.replace('jpg', 'mat'))

#Load segmentation from sixth human
segm=gt['groundTruth'][0,5][0][0]['Segmentation']
plt.imshow(segm, cmap=plt.get_cmap('hot'))
plt.colorbar()
plt.show()

#Boundaries from third human
segm=gt['groundTruth'][0,2][0][0]['Boundaries']
plt.imshow(segm)
plt.colorbar()
plt.show()

```

Feel free to use/implement any evaluation strategy for your segmentation function. **DO NOT use the actual BSDS500 evaluation strategy (we will be there next week)**. Just use a simple strategy that can yield a reasonable evaluation of your segmentation function. Remember there are multiple ground truths for any given image. 

## Your turn

The report for this laboratory must include:

-   Small (Max. one paragraph per method) description of the clustering algorithms.

Segmentation parameter tuning and Image preprocessing

-   Did you have to scale (up or down) the value of any of the channels (r,g,b,x,y). why?
-   Did you have to rescale the images, why? Does it affect the result?
-   The hyperparameter 'numberOfClusters', is probably the most important parameter in this problem, how can you choose it?

Evaluation
-  How can we evaluate a general segmentation problem? How can we handle the multiple ground truths?
-  What evaluation strategy did you choose, why? What would be its shortcomings, if any?
-  Using only your evaluation strategy, how do your segmentation methods perform on this sample set?

Discuss the results
-  Does it make sense to use other color spaces different from RGB? Why?
-  What segmentation method or color space seems to yield the best result, can you give some insight on why?
-  What are the limitations of the method? (again CPU and RAM memory are well known constraints, try to go further!)
-  Do you think any of the channels seems to be most discriminative?
-  Overall, what seem to be the fail conditions of the implemented methods?
-  How could you improve your evaluation strategy? Are there any drawbacks?
-  Finally,  how could you improve your best method?

The report should have max 5 pages (**English**), if it is necessary use any additional pages for references and images **only**. Use the standard CVPR sections: abstract, introduction (again, be concise), methods, results, conclusions and references. There is no need for an 'State of the art' section.

## Deliverable (python users)

**You will end with at least 2 `.py` scripts *i.e.*,  Segment.py (or however you have named your function) and main.py.** 

*main.py* must be an executable file with the #!.. at the beginning of the file. This opening line must be consistent with the python version you are using.

*main.py* must accept the image file, color space, number of clusters, and method to perform clustering, by arguments like this:

Python users: `./main.py --img_file=xxx --color=xxx --k=xxx --method=xxx`

IPython users: `./main.py -- --img_file=xxx --color=xxx --k=xxx --method=xxx` (check the double line).

For this purpose, you might need `argparse` module:

```python
def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def groundtruth(img_file):
    import scipy.io as sio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,5][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    
def check_dataset(folder):
    import os
    if not os.path.isdir(folder):
        # Download it.
        # Put your code here. Then remove the 'pass' command.
        pass

if __name__ == '__main__':
    import argparse
    import imageio
    from Segment import segmentByClustering # Change this line if your function has a different name
    parser = argparse.ArgumentParser()

    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)
	
    opts = parser.parse_args()

    check_dataset(opts.img_file.split('/')[0])

    img = imageio.imread(opts.img_file)
    seg = segmentByClustering(rgbImage=img, colorSpace=opts.color, clusteringMethod=opts.method, numberOfClusters=opts.k)
    imshow(img, seg, title='Prediction')
    groundtruth(opts.img_file)
```



## Tip

- `skimage` is the module you are looking for in order to move across space colors. For instance:

```python
from skimage import io, color
img = 'BSDS_small/train/22090.jpg'
rgb = io.imread(filename)
lab = color.rgb2lab(rgb)

#try to visualize lab image
plt.imshow(lab) # ???
plt.show()

def debugImg(rawData):
  import cv2
  toShow = np.zeros((rawData.shape), dtype=np.uint8)
  cv2.normalize(rawData, toShow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  # cv2.imwrite('name', toShow)
  plt.imshow(toShow)
  plt.show()

debugImg(lab) #Different enough?

```

## Deadline 
**March 7, 11:59 pm**.


