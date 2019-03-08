# Berkeley Segmentation Dataset and Benchmark (BSDS500)

In this lab you will be testing your previously developed segmentation functions in a real and large segmentation database, the BSDS500. To add some extra motivation, you will also be comparing your methods against that of your teacher, can you beat him?

## Resources

This lab uses resources from the [Berkley Computer Vision Group](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), you might want to check their website.

## Download data

All the data you need for this lab is already packed on the course server and can be downloaded at:

- http://bcv001.uniandes.edu.co/BSDS500FastBench.tar.gz

It contains both the images and the fast benchmark code.

**Note**: You probably don't want to add all these files to your repository, most of them are images, and they are really big.

## Data description

The folder contains the following directories:

- bench_fast: Approximate, faster version of the original BSDS benchmark code.
- BSD500:
- Data: Images along with their manual segmentations (ground truth).
  - ucm2: Precalculated results of the *Ultrametric Contour Map* segmentation algorithm.
  - Documentation: Paper describing the UCM algorithm.
- Grouping: Implementation of the UCM algorithm (for those rad students that get offended at the very mention of precalculated results ;) ) 

## BSDS Benchmark

Before you proceed keep in mind that, just like with the texton lab, this benchmark is **quite computationally expensive**. Plan ahead and be mindful of the limited resources on the course servers. Unlike Lab05, the computation time in the 'test set' is as expensive as in the train or val set.

The file ``bench_fast/test_benchs_fast.m`` contains two examples on how to use the fast benchmark functions with two different types of output images. You might want to focus on the one labeled **%% 4**, It is the morphological version for: all the benchmarks for results stored as a cell of segmentations", as this one works directly with segments.


## Your Turn

There is little coding to do in this lab, most of the 'heavy lifting' was already done in Lab06. To start, take a look at the files at (./BSR/bench_fast/data/segs), the ultimate goal of your code is to produce **matlab data files just like these ones**. They are just cell arrays, where each element is a segmentation of the original image calculated with a different K; Be consistent across all images, that is, the N-th element on the array for any image should be calculated with the same K.

### Start with your best segmentation method

Choose two of the functions you developed during the **Lab06**, use those that had the best performance. Feel free to modify or enhance any algorithm based on your findings for the last lab. If you think any of the functions is too computationally expensive, this is the moment to make some adjustments. Remember the benchmark must be run with **all 200 images**.

Now, adjust the function you designed in the past lab so that it works on the BSDS Benchmark. Essentially you must be able to process a full set of images (train/val/test) and then write the segmentation results in the same format as those in '/bench_fast/data/segs'. 

There is, at least, one hyper-parameter for your segmentation method: K (number of clusters). Use the ``train``  and ``val`` sets to explore the best set of values for K and any other hyper parameter.

### Testing your method

Once you get the segmented images in the proper format, run the benchmark for your selected methods, also run the  benchmark for the provided UCM segmentations (at BSDS500/ucm2). Use **only** the ``test`` for comparisons.

Do not forget to use different thresholds values to generate a nice curve. See http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bench for an example output.

A summary of the steps required is:

- Apply your function to each of the images in the test dataset
- For each image, create a `.mat` file that contains a cell array, which contains several arrays representing the output of the function for different values of the parameter (k). Look at the bench_fast/data/segs folder for examples.
- Run the `allBench_fast` function specifying the folder with the .mat files as `inDir`. See the second example in the `test_benchs_fast` file.
- Generate the plot using the code from the `plot_eval` function.



### Report

The report for this laboratory must include:
- A brief description of the selected segmentation methods, why did you choose them?, did you make any modification, or enhancements? Why?
- What are the hyper parameters of these methods, what do they mean? How could you choose them? 
- What does the precision-recall curve tells us? What would be the ideal result?
- What can you say about the ODS, OIS, F-max and Cover metrics?
- Results for the BSDS benchmark (**full test set**) for the three methods.  Do not forget to include a graphic where you compare the curves you generated for the three methods (UCM + yours).
- Among the methods you developed, which one works best, why?
- Did you beat Pablo, No? Why?
- On the last Lab, you made a simple comparison of segmentation methods, does the BSDS benchmark has similar results with yours (i.e. your best algorithm is still the best)? Why? Or why not?
- What seems to be the limitations of the algorithms you developed? Again execution time and resource usage are not our main concern.
- Do you think the errors on your segmentation methods follow any pattern(s)?
- How could you improve your algorithms? Think big, be creative, we are in the what if domain.



### Additional Resources

- [Segmentation metrics](https://www-sciencedirect-com.ezproxy.uniandes.edu.co:8443/science/article/pii/S0047259X06002016).
- [Figure example](http://cs.brown.edu/courses/cs143/2011/results/proj2/lbsun/).
- [Matlab debugging](https://www.mathworks.com/help/matlab/ref/dbstop.html).



### Deadline
**March 14, 11:59 pm**