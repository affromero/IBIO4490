# Logistic Regression

Before going deep with Neural Networks we need to ground the concepts a little bit. In this lab we will introduce the very basic Machine Learning concept of Logistic Regression in order to classify happy faces on the [FER2013](https://www.kaggle.com/deadskull7/fer2013/version/1]) dataset. 

<p align="center"><img width="60%" src="https://www.researchgate.net/profile/Martin_Kampel/publication/311573401/figure/fig1/AS:613880195723264@1523371852035/Example-images-from-the-FER2013-dataset-3-illustrating-variabilities-in-illumination.png" /></p>

## Machine Learning Concepts

### Loss

Every machine learning approach either supervised learning, unsupervised learning or even reinforcement learning deal with a cost function. This tells us how good the system is working under certain conditions. Choosing the right loss function is **extremely** important. Traditional (classical (?)) methods only works with one single cost function. There are plenty of loss functions out there including but not limited to:

- `Linear regression`. Purely regression, also known as *L2 loss*.
- `L1 loss`.
- `Logistic regression`. Binary classification problems, also known as *Cross Entropy*. *Binary Cross-Entropy* for Multi-label datasets. [link](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc). 
- `Softmax Cross-Entropy`. Generalization of the Logistic regression for many classes. 

Propaganda: If you ever heard of Generative Adversarial Networks (if not, you will, trust me), one of the success about it is that they work with an ensemble of more than 3 cost functions (I use 7 for my current project).

### Optimizer

Wait, what? 

An optimizer is basically the **learning** pathway. You may *learn* C++ either slowly, very quicky or in an **optimal** way. 

Disclaimer: There is no such a fast way to learn C++.

![Resultado de imagen para how to learn c++ fast](https://camo.githubusercontent.com/8cfb8cdc747507a3b2620e73b5226385b78ad94c/687474703a2f2f6162737472757365676f6f73652e636f6d2f7374726970732f6172735f6c6f6e67615f766974615f6272657669732e706e67)

Then, there are several optimizers out there. The most famous and popular one is [Stochastic Gradient Descent](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0) (SGD). Recommended [video](https://www.coursera.org/lecture/machine-learning/stochastic-gradient-descent-DoRHJ) by the Machine Learning rockstar Andrew NG.

There are some hyperparameters such as *Momentum* and *Learning Rate* (get used to this one) that tell the optimizer how quickly it must move. Later, you will see there are other hyperparameters related to regularization and so on. 

![](https://jed-ai.github.io//images/python/py1/fig5.gif)





Once you set the loss function and the optimizer, hands on it. 

## Thinking Time - FAQ

![](https://media.giphy.com/media/SabSYEpsVh0di/giphy.gif)

- `What is the difference between this and the Statistical course I took a couple of semesters ago`? NONE, it just has a cooler name :), Machine Learning sounds better than Statistics. 
- `How is this even related to Deep Learning?` Pretty much, just be patient. There is a huge hype around deep learning. 
- `When will I see a Neuron?` Never. Just imagine each number in the Support Vector Machine model is a neuron that *thinks* for itself. It does not make sense right? I know, either your mind is about to blow or you are very disappointed, or both. Withdrawal time is no longer available, I am sorry. 
- `Is it always better to use Deep Learning over LinearRegression/LogisticRegression?` Well, is it wise to drive a Ferrari in a dessert? You can and somehow manage to work it out, you will look like a not-very-smart person though. 



## Your turn

- First of all, download the dataset and decompress it.

- Familiarize with `main.py` script. It reads the data from the *csv* file, and trains a logistic model. Play (wisely) around with the parameters. 

- Implement both the `plot` and `test` function, that is plot train and test losses, and evaluate the test set (ACA, PR Curve and F1 measure).

- Do your training look like this? Be aware you select your best model based on the **validation** scores, not the **test** set - it is highly important. To this end, you **must** split (as you wish) the train into train and val and the test remains untouched. 

  <p align="left"><img width="30%" src="https://forums.fast.ai/uploads/default/original/2X/d/db413a396fe0d3555f15e78e05bb36a2141bb8a4.jpg" /></p>

- Study the behavior of these hyper-parameters: 

  - `batch_size`: Does it converge when is it too low? Is it better when is it high? Is there a Plateau region for high numbers?
  - How can you relate the `learning rate` with the training time and the `convergence` of the model. Is there an optimal value?
  - Would it be wise to re-scale the `learning rate` when the loss is not converging too much? (Divide the LR by some number to fine the results).

- What would you change to deal with all the emotions (7) using **one** single loss function? Explain the theory behind and train the whole system (create a new script `main_emotions.py` for that purpose). Tip: `Softmax Cross-Entropy`. 

### Deliverable

- `main.py` 
  - `python main.py` runs the whole system, that is train and test.
  - `python main.py --test` only test the system, that is to depict the PR curve, F1 and normalized ACA. 
  - `python main.py --demo` use images in-the-wild to tell how good the system is classiying. 
- `main_emotions.py`
  - `python main_emotions.py` runs the whole system, that is train and test.
  - `python main_emotions.py --test` only test the system, that is to depict only the normalized ACA. 
  - `python main_emotions.py --demo` use images in-the-wild to tell how good the system is classiying.
- `LastName_LogReg.pdf`
- `images` folder with your images-in-the-wild. 

Disclaimer: Your code must download the dataset internally. I **strongly** recommend Python3.7.