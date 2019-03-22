
# HOG Face Detection
Just like in the last laboratory, we will be using another classic computer vision strategy, namely, the multi-scale HOG descriptor for object detection. Again our codebase will be the VLFeat Library. This time we will switch our attention from generic objects to a far more constrained detection target.

For this Lab, we will use a small subset of a modern database the [WIDER FACE: A Face Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). This dataset is **very challenging** with over 393,703 and contains face instances from several every day scenarios like public events, family meetings, common leisure activities, among many others.

## Resources

This lab uses resources from the [Multimedia Laboratory, of the Chinese University of Hong Kong](http://mmlab.ie.cuhk.edu.hk), you might want to check their website.

To better approach the problem (read to get results above 0.0) we will operate over a very small subset of WIDER FACE. We will work only on face instances whose area is larger than 80x80 pixels, avoiding those who are rather small and/or have a significant amount of blur.

Once again most of the technical heavy lifting is already done, as the codebase for this Lab is provided and thoroughly explained by the [Visual Geometry Group at Oxford](http://www.robots.ox.ac.uk/~vgg/practicals/category-detection/). Read their tutorial and make sure that the technical aspects of their implementations are clear.

## Data 
The subset of WIDER FACE that we will use in this lab can be downloaded at:

http://157.253.63.7/lab9Detection.tar.gz

There are 3 directories in the download:

- Train: Set of train images that contain at least 1 face (as by the specs of our subset).
- Train Crops: Resized face crops with roughly the same size.
- Val: Set of validation (testing) images that contain at least 1 face (as by the specs of our subset).


## Evaluation
As we work on a very small subset of the original dataset, the standard evaluation code is not appropriate, therefore we will be using a modified version of the original evaluation code that applies only to the subset we selected. This modified version can be obtained at:

http://157.253.63.7/eval_tools_Lab9.tar.gz

After download, adapt the hardcoded paths to your environment, they are easy to spot as they all begin with '/home/afromero'

To use the evaluation code your results must be in a specific format, the prediction for each image should be included in a file as follows :

```
0_Parade_Parade_0_102   -> image name, ignore extension
2                       -> number of face detections on image
499 176 59 59 0.25      -> x coodinate of top left corner, y coodinate of top left corner, width, height, confidence
568 625 84 84 0.5
```

## Your Turn

Create a multi-scale HOG detector for faces using the provided dataset. Then evaluate your detector on the test set using the provided script

We are now far more experienced in the Computer vision world, so this time you will get no further guidelines. Moreover you are free to apply any modification, extension pre/post-processing to the base algorithm or data, Notice this means you can even add more images from the original dataset **on the train phase**. 

In other words **Everything goes** as long as:
- You don't cheat.
- Test set remains unaltered
- **The core of your strategy remains a multi-scale HOG detector**.

## Report 
The report for this laboratory must include:
- A brief description of the multiscale multi-scale HOG strategy, why can you apply it to a detection problem?
- Can you identify any hyper parameter on the multi-scale HOG?, what is it useful for?
- How can you evaluate a general detection problem? 
- Overall description of your strategy including any modifications/enhancements you applied to it.
- Your results on our test subset of the wider-face dataset.
- What seems to be the limitations of the strategy  you developed?
- Do you think the false positives follow a pattern?
- Do you think the false negatives follow a pattern?
- How could you improve your algorithms?


## Extra credit 1
Ever heard of the [viola-jones algorithm](http://www.vision.caltech.edu/html-files/EE148-2005-Spring/pprs/viola04ijcv.pdf)?. It is classic boosting strategy first presented for real-time face detection, it can be easily extended to other detection problems. 
There are lots of implementations around the web, pick the one you like the most and get results on our test set. Include in your report the **quantitative** comparison against multiscale-hog, which one works best? any idea why?

## Extra credit 2

Our good man waldo is hidden somewhere in the dataset, can you find him? Tip use this mugshot to find him. gl hf

<img src="https://pbs.twimg.com/profile_images/561277979855056896/4yRcS2Zo.png" alt="Waldo" width="250" height="250">


## Deadline
**4 April 11:59** As usual upload your report on github. Only submit the **pdf** and the code. <**forget the ANSWER folder**>. It is important that you remember to depure space on your repo since I am having a bad time cloning all of them. 
