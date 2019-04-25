# Convolutional Neural Networks for Attribute Classification
In this lab we will be exploring the use of Convolutional Neural Networks (CNN) for image classification. We will start from the very beginning, that is, we will design and train a CNN from a random weight initialization.

**Note, while executing the code does not take that long as in other labs (or maybe it does), you will have to discover an appropriate network by trial and error, this might be time consuming, plan ahead and don't get discouraged by the negative results.**

## 1. Resources

Following the last lab, in this lab feel free to use any combination of layers that comes up with a decent performance. Then, you can build over your network (adding layers, non-lineartities, etc) and improve the performance. 

If you wish, you can use the traditional [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) or **any** other architecture you may find appropiate. 

If you decided to go through AlexNet, take a time to figure out the AlexNet architecture and their groundbreaking approach (at the time), or at least get a nice grasp about their contribution. 

You can easily get the entire network in the Pytorch repository: [alexnet-pytorch](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py). Note that this implementation uses a non-linearity that is different from the original paper. 

## 2. Data

<p align="center"><img width="40%" src="http://mmlab.ie.cuhk.edu.hk/projects/celeba/intro.png" /></p>

For this lab we will use the thrilling [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. It consists of roughly 200k images of celebrities (?) and it is labeled with 40 different attributes (not-mutually exclusive) such as wearing eyeglasses, smiling, arched eyebrows, male/female (do not judge about binary genders), pale skin, young/old, etc. Here I show you the full labels:

                '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                'Wearing_Necktie', 'Young'
For simplicity, we will train our models for only 10 out of them: 

                'Eyeglasses', 'Bangs', 'Black_Hair', 'Blond_Hair',
                'Brown_Hair', 'Gray_Hair', 'Male', 'Pale_Skin', 'Smiling',
                'Young'
<p align="center"><img width="60%" src="http://mmlab.ie.cuhk.edu.hk/projects/celeba/overview.png" /></p>

You can download the dataset officially from the website link or from [kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset). Nevertheless, for all of you lazy people, you can get it at this location: `/media/user_home2/vision/data/CelebA`.

**You can either train your model using raw images or extracting the face with face detector first.** 

### 2.1. Optional

If you decide to go through off-the-shelf networks, bear in mind about the special input sizes and normalization for those networks for instance, 227x227 for AlexNet, 224x224 for [VGG](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py), 299x299 for [ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). Take a look at the pytorch [model-zoo](https://github.com/pytorch/vision/tree/master/torchvision/models). 

**Do not waste too much time trying different sizes (if you want only two experiments with it). Focus on the network.**

**It is important to bear in mind that the full dataset (without random sampling) is about 100k images. you can subsample the dataset however you want. Bear in mind the concept of overfitting and underfitting** . 

### 2.2. Data Augmentation

Pytorch includes different data augmentation [options](https://pytorch.org/docs/stable/torchvision/transforms.html), take a time ti figure all of those down. 

## 3. Recommended Resources
This [paper](https://arxiv.org/abs/1407.1610) will give you further insight on what to try while training a CNN. It is certainly not a technical tutorial, but I strongly recommend to read it before you start designing CNN architectures.

## 4. Your turn

Design a Neural Network to classify this facial attribute dataset. Just like in Lab 10 you are on your own. 

The one requirement is to use **only a CNN**, that is you are not allowed to apply any pre/postprocessing and other vision or learning strategies are forbidden.

## 5. Report
The report for this lab should be brief (no more than 4 pages, you can still use additional pages for figures and references). It must contain the following information:

- First of all, how would you tackle this problem without CNNs?

- A description of your network, and the ideas you tried to implement on that design. You should come up with at least **one** architecture by your own. 
- What challenges did you face while designing the architecture? How much you had to change your original design until it worked?
- Does the use of jitter helps?
- Ablation studies, we will try to explain why the network works by removing some layers from it, how does each removed layer affect the performance? What does it tell about your architecture?
- The results of your network in train and **test** sets.

Do not forget to upload a *train.py*, *test.py* and *demo.py*. Explanation about those files is not required at this point. One subtle difference is that the *test.py* must depict the description of the network, that is by running this script, it must displays the network, the number of parameters and the output size of each (exception for non-linearities and normalization layers).

## 6. Due Date:
**April 25 2017 11:59pm**.

## 7. BONUS, The CelebA Recognition Challenge 
We will be holding our 'CelebA classification challenge', like most real-world challenges you are free to use **any strategy (cheating is not a valid strategy!)** to produce the better classification over the test set of our modified CelebA database. Unlike real world challenges, you cannot develop a joint solution with another group, any such submission will be disregarded. 

Your Submissions will have a standard format (just like in Lab 11): *.txt* extension. 

The challenge server is available [here](http://bcv001.uniandes.edu.co/contest/). As before, it will evaluate the per class accuracy, precision, recall, and F1-score. A leaderboard is also available on the same server. 

As this extra credit requires a lot of effort, there will be a special bonus. The best submission replace again his/her worst grade and the second two submissions will get a +2.0 that can be added to any one of their Labs grades. 

Finally, to add some extra motivation, your lab instructor will also be part of this challenge (he will not cheat), can you beat him?

![](https://media.giphy.com/media/26BRzQS5HXcEWM7du/giphy.gif)


