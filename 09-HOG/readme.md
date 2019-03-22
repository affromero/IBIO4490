
# HOG Face Detection
Just like in the last laboratory, we will be using another classic computer vision strategy, namely, the multi-scale HOG descriptor for object detection. Again our codebase will be the VLFeat Library. This time we will switch our attention from generic objects to a far more constrained detection target.

For this Lab, we will use resources from the [UTexas](http://vision.cs.utexas.edu/378h-spring2017/assignments/a5/A5.html) Computer Vision course. You may want to check their website. We are going to detect faces using the [Caltech Web Faces](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/) dataset. It contains roughly 6,713 cropped 36x36 faces for training and 130 randomly sized images with lot of faces. 

Creating the sliding window, multiscale detector is the most complex part of this project. It is recommended that you start with a *single scale* detector which does not detect faces at multiple scales in each test image. Such a detector will not work nearly as well (perhaps 0.3 average precision) compared to the full multi-scale detector. With a well trained multi-scale detector with small step size you can expect to match the papers linked above in performance with average precision above 0.9.

## Resources

http://bcv001.uniandes.edu.co/LabHOG.zip contains the resources for this lab. There are 2 directories there:

- Data:
  - caltch_faces: 6,713 cropped 36x36 face images.
  - test_scenes: 130 images for testing.
  - train_non_face_scenes: Negative samples.
  - extra_test_scenes: It will be updated with **YOUR** pictures for qualitative result and analysis purposes (I will take pictures of you :| ). 
- Code:
  - `main.m` The top level script for training and testing your object detector. If you run the code unmodified, it will predict random faces in the test images. It calls the following functions, many of which are simply placeholders in the starter code.
  - `get_positive_features.m` (**you code this**). Load cropped positive trained examples (faces) and convert them to HoG features with a call to vl_hog.
  - `get_random_negative_features.m` (**you code this**). Sample random negative examples from scenes which contain no faces and convert them to HoG features. You can even add more data here. 
  - `classifier training` (**you code this**). Train a linear classifier from the positive and negative examples with a call to vl_trainsvm.
  - `run_detector.m` (**you code this**). Run the classifier on the test set. For each image, run the classifier at multiple scales and then call non_max_supr_bbox to remove duplicate detections.
  - `evaluate_detections.m`. Compute ROC curve, precision-recall curve, and average precision. You're not allowed to change this function.
  - `visualize_detections_by_image.m`. Visualize detections in each image. You can use visualize_detections_by_image_no_gt.m for test cases which have no ground truth annotations (e.g. the class photos).




## Evaluation
Inside the code there is an evaluation methodology using Average Precision and other metrics you may want to use for you analysis. To this end, the code does not expect any `txt_file` with the predictions, instead it uses the SVM trained parameters (w, and b) to compute the detection online.

## Your Turn

Create a multi-scale HOG detector for faces using the provided dataset. Then evaluate your detector on the test set using the provided script

We are now far more experienced in the Computer vision world, so this time you will get no further guidelines. Moreover you are free to apply any modification, extension pre/post-processing to the base algorithm or data, Notice this means you can even add more images from the original dataset **on the train phase**. (if you do this, you must analyze both cases i.e., with and without more data). 

In other words **Everything goes** as long as:
- You don't cheat.
- Test set remains unaltered
- **The core of your strategy remains a multi-scale HOG detector**.

## Report 
The report for this laboratory must include:
- A brief description of the multiscale multi-scale HOG strategy, why can you apply it to a detection problem?
- Can you identify any hyper parameter on the multi-scale HOG? What is it useful for?
- How can you evaluate a general detection problem? 
- Overall description of your strategy including any modifications/enhancements you applied to it.
- Show and discuss the results of your algorithm.
- Show how your detector performs on additional images in the `data/extra_test_scenes` directory.
- Include the precision-recall curve and AP of your final classifier and any interesting variants of your algorithm.
- What seems to be the limitations of the strategy you developed?
- Do you think the false positives follow a pattern?
- Do you think the false negatives follow a pattern?
- How could you improve your algorithms? 

## Face detection contest

> There will be extra credit and recognition for the students who achieve the highest average precision, whether with the baseline classifier or any bells and whistles from the extra credit. You aren't allowed to modify `evaluate_all_detections.m` which measures your accuracy.


## Extra credit 1
Ever heard of the [viola-jones algorithm](http://www.vision.caltech.edu/html-files/EE148-2005-Spring/pprs/viola04ijcv.pdf)?. It is classic boosting strategy first presented for real-time face detection, it can be easily extended to other detection problems. 
There are lots of implementations around the web, pick the one you like the most and get results on our test set. Include in your report the **quantitative** comparison against multiscale-hog, which one works best? any idea why?

## Extra credit 2

Our good man waldo is hidden somewhere in the dataset, can you find him?. Tip use this mugshot to find him. gl hf

<img src="https://pbs.twimg.com/profile_images/561277979855056896/4yRcS2Zo.png" alt="Waldo" width="250" height="250">


## Deadline
**31 March** Upload your report on github as follow:

- code (Folder that contains everything you need to run the code. Be aware of internally create a symlink to MY **vlfeat** and compile it as well).
- images (Folder that contains test and demo sets)
- demo.m (It shows qualitative results over one randomly picked from `images/demo` pictures. **Be aware of the SVM pretrained weights**.).
- test.m (It depicts figures for the entire test set with respect to the groundtruth).
- Romero_HOG.pdf (Change to your lastname. Each student **should** upload it accordingly in his/her corresponding repo)
