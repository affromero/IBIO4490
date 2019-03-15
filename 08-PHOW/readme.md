
# PHOW Image Classification

In this Lab we will be exploring a classic computer vision strategy for object recognition namely the "Pyramid histograms of visual words" or "PHOW". Unlike most Labs there will be little coding to do, as we will use the VLFeat Library, originally developed by Andrea Vedaldi and Brian Fulkerson. Moreover, we will also take a peek a two essential datasets for the object recognition task. The classic (read old) caltech 101 dataset, and the ImageNet dataset, currently the de-facto standard for object recognition.

## Setup

First download the [vl_feat](http://www.vlfeat.org/index.html) library. After downloading and uncompressing the folder, open matlab and run the following command:

```matlab
run('**VLFEATROOT**/toolbox/vl_setup')
```

This simple installation procedure should work for most. For detailed information and troubleshooting see [vl_feat matlab toolbox installation](http://www.vlfeat.org/install-matlab.html).

## Example

The vl_feat library already includes an excellent example on how to use PHOW on the caltech 101 dataset see [phow_caltech101](http://www.vlfeat.org/applications/caltech-101-code.html).

Analyze the script and do your best to understand what it does. If you want, run it using checkpoints to further see how it works. Do not forget to take a close look at the results and its file output, be sure you understand the former, and you know what is being stored for the later. Tip: there are relevant parameters ‘hidden’ well beyond the first few lines of the script.

Experiment further with the script to try to find out the following

- How does the problem changes when the number of categories increases? 
- How does the problem changes when the size of the train set changes?
- How does the result changes when the number of dictionary words changes?
- How does the result changes when SVM C parameter changes?
- How does the problem changes when the spatial partitioning changes (``conf.numSpatialX/Y``)?

Try also to figure out how these settings change the memory and CPU time required, we are switching to a larger dataset.

## Your Turn

Use PHOW to train and test a classifier in a **very small** subset of the [image-net](http://www.image-net.org) database.

- http://bcv001.uniandes.edu.co/imageNet200.tar

**This is a large file (~2.2GB)**, so it would be best to download it using ``wget`` in the course server

Adapt the script from the example to work the train set on this new dataset. What performance do you get with default params? How does it compare to the results from caltech-101? Any idea what causes the differences?.

Finally develop a small classification function which allows you to use your best model on the test Set.

Tip: Using 100 images from 200 classes will add a significant load on the course server, and you probably will run out of memory. You can, however, reduce the number of images per class and train with the whole class-set.


## Report

The report for this laboratory must include. 

- Short description of the Caltech 101 and Imagenet dataset (max 1 paragraph each). Depicts pictures for both of them, show how easy/hard these problems are. 
- Short description of the PHOW strategy (max 2 paragraphs). 
- What is the difference between PHOW and SIFT? Is it PHOW scale invariant? Why/not?
- What is the difference between PHOW and Textons? Is it worth it?
- What are the most relevant hyperparameters for the PHOW strategy. Did you found any other relevant parameters inside the script?
- How can you choose the best set of hyper parameters for Caltech 101 and imagenet set? What are their values?
- Evaluate the performance (ACA) of the classifier in the train set of imagenet and Caltech 101 and imagenet. Report a single number for the whole set. Why is there such a big difference?
- Performance (ACA) Using your classification function on imagenet test, again a single figure.
- What seem to be the ‘easy’ classes, what seem to be the hardest? Why? Show pictures, analyze model behavior. Not stay just with "The easyest are cars, the hardest flowers." Yeah, and? Elaborate. 
- What are the challenges of the system? (**Go beyond timing and hardware resources**).
- How could you improve your results, go beyond a better parameter set/exploration as this is rather obvious. Think big, be creative, again we are in the what if domain. 

Skip informal sentences such as:

- We now proceed to evaluate using the matlab function...
- These results are terrible... (Of course they are, that is the very purpose of the lab. BUT WHY ARE THEY SO TERRIBLE?)
- We tried this experiment and it did not work...
- Because of the timing we were not able to run a good batches of experiments...
- REMEMBER: Imagine you are presenting a peer-reviewed paper, would you write as you speak? Take a look on *google scholar* to familiarize with the writing style in the Computer Vision field. There is a very/special standardized style. Get used to it, you will need it for the project. (Thank me later :) ).

Also:

- Computer Vision papers with no pictures it is like the Bolivian Navy. That simply does not make any sense.
- Captions must be self-included, *i.e.*, only by reading the pictures/tables the reader must get a glimpse about the whole paper without getting into the details. 
- Figures and Tables must be mentioned in the article. If you put tons of pictures and you never discuss about them, it is like they are not there, and what is worst, you will not be taken seriously. 
- Avoid to use figures that must be zoomed in. It is *already* very cumbersome to figure out what a figure is trying to depict without the proper caption. 
- Avoid to run as many experiment as you can, then put 20 figures, somehow say which one is the best and that is the whole discussion. It is better to depict one single picture but comprehensively discussed than several pictures with almost none analysis.
- It is very acceptable, even encouraged to bring theoretical properties or hypothesis into the analysis (L from the LAB space color gives better information for the segmentation, bla bla bla). What it is not acceptable is not to prove or not to refute those theoretical guidelines. Again, your main function is to always wonder, **why does it work? why it doesn't work?** Don't keep those thoughts only to your mind, use the paper for that, share them with the world. Explore what you have done, ask yourself "does it makes sense what I have just done?"
- PLEASE, use subsections. One full section with ten paragraphs is very difficult to follow if you do not do it perfectly. Instead, you can easily separate each couple (several) paragraph by subsections which gives clarity and elegance. 
- PLEASE, PLEASE, PLEASE (Almost crying) if you want to look like a high school teenager writing a paper, use screenshots. There is nothing less professional than that. You think it is non-sense but it does not. One of the reason we all use Latex is its excellence visual aesthetic. Every time you upload a screenshot to Overleaf, someone dies.  

## Deadline
**March 24 11:59pm**.



