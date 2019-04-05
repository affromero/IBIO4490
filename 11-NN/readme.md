# Neural Networks Tutorial

In this lab we will introduce the very basic concepts of neural networks using the classic [MNIST](http://yann.lecun.com/exdb/mnist/]) dataset. Just like the very first Labs, this is a tutorial, hence there are no deliverables or deadlines.

<p align="center"><img width="40%" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/440px-MnistExamples.png" /></p>


## Resources

This lab uses the Facebooks's own machine intelligence library [PyTorch](http://pytorch.org/). You might want to check its website.

<p align="center"><img width="40%" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" /></p>

PyTorch is more than a simple Deep Learning Framework, it is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system


## Why PyTorch (Propaganda time)

### Python First
Most current frameworks like Caffe or Tensorflow define tools and bindings around static C/C++ routines, sometimes in a Un-Pythonic way. PyTorch aims to put Python first and offer a simple and extendible syntax, such that is not differentiable from that of classical libraries such as Numpy/Scipy or scikit-learn.

### More than a Neural Network library
PyTorch is a general tensor/matrix library that allows to perform operations on a GPU without any hassle nor any complicated syntax. It is just as simple as invoking the ``.cuda()/.cpu()`` function, and all your operations can reside on the GPU or the CPU transparently. PyTorch is more close to a GPU Numpy, allowing to perform calculations beyond the Deep Learning realm.

### Debug as you execute
Because PyTorch instructions can be executed directly on a Python interpreter, you can call each instruction and see its result synchronously, differring from other asynchronous frameworks on which you must compile a model and then execute it to see if it is working properly. Say goodbye to execution engines and sessions.

### Autograd
With respect to Deep Learning applications, PyTorch defines a backpropagation graph on which each node represents a mathematical operation, whereas the edges represent the sequence and forward relation between them. Different from TensorFlow, PyTorch defines dynamic graphs defined at run-time rather than at compile-time, allowing to change a networks' architecture easily and with minimal time overhead.

PyTorch uses a technique called reverse-mode auto-differentiation, which allows you to change the way your network behaves arbitrarily with zero lag or overhead. It's state-of-the-art performance comes from several research papers on this topic, as well as current and past work such as autograd, Chainer, etc. This is actually one of the most efficient and fast current implementations so far.

<p align=center><img width="80%" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/dynamic_graph.gif" /></p>

## Hands on it

PyTorch **is already installed** at the course servers. Unless you have a Cuda capable **Nvidia** graphic card, you must use the course servers. Otherwise, be patient. Nevertheless, you can install PyTorch (GPU enabled | cuda 9.0) either using `conda`, `pip` or from `source`:

```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

```bash
pip3 install torch torchvision
```

```bash
# Follow instructions at this URL:
https://github.com/pytorch/pytorch#from-source
```

The full scripts for this Lab are available [here](fcMNIST.py), [here](fcMNIST2.py) and [here](convMNIST.py), respectively, no comments though.

### Handwritten digits recognition using Neural Networks
The data for the lab is available online and can be automatically downloaded using pytorch, it is as simple as:

### torchvision package
**torchvision** is a popular pytorch package that consists of popular datasets, model architectures, and common image transformations for computer vision.

```python
#Utilities for MNIST dataset on pytorh
from torchvision import datasets, transforms

# Load pre-shuffled MNIST data into train and test sets
train = datasets.MNIST('data', train=True, download=True)
test = datasets.MNIST('data', train=False)

# Inspect the size of the train data
train.data.shape
# Inspect the size of the train labels
train.targets.shape

#From Tensors (torch) to Numpy
numpy_data = train.data.numpy() #As simple as it is

#From numpy to Tensors
import torch
data_train = torch.from_numpy(numpy_data)

#From CPU to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(type(data_train), data_train.device)
data_train = data_train.to(device)
print(type(data_train), data_train.device)

#From GPU to CPU just change to(device) with cpu() e.g., data = data.cpu()
```

MNIST images are very small (28x28) grayscale images containing a single handwritten digit, you can render the first image in the training set using the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

#5 is the first image on the train set
img = plt.imshow(data_train.cpu().numpy()[0])
plt.show()
```

We will first approach the recognition problem using fully connected networks, therefore our first step will be to turn images into an array of 1x784


```python
X_train = train.data.numpy()
X_test = test.data.numpy()
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

# Inspect the new size of the train data
print ('X_train.shape ', X_train.shape)
print ('X_test.shape ', X_test.shape)

```

## 1. Small network
Now we can build a Neural Network to classify the MNIST data. As first example, we will use a minimal 1 layer 1 fully connected neural network


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

batch_size=128

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
data_train = datasets.MNIST('data', train=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

data_test = datasets.MNIST('data', train=True, transform=transform_train)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

#Define a new neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 1) #Add 1 fully connected layer with 1 neuron

    def forward(self, x):
        x = self.fc(x)
        return x

#display the model summary, check the number of parameters
model = Net()

#Use GPU - IT IS NOT REQUIRED: model = model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 

num_params=0
for p in model.parameters():
    num_params+=p.numel()
print("The number of parameters {}".format(num_params))

#Actually getting the param/layer weights
print(model.state_dict().keys())
print(model.state_dict()['fc.weight'].shape)

```

Before we start the optimization process, take a time to think about the network, how many parameters does it have?, what kind of operation does the neuron?

Now start the optimization process, using Stochastic gradient descent (sgd), a learning rate of 0.01, no decay and a momentum of 0.9. since we have a single neuron we have to relax the classification problem into an estimation one, thus, we use a mean squared error loss function (a bit of a cheapshot, but is the only way to train a network of this size for a classification problem):


```python

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Optimization params
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

#Loss
Loss = nn.MSELoss()

#Start training process
epochs=20

import tqdm
import numpy as np
model.train()
for epoch in range(epochs):
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch: {}".format(epoch)):
        data = data.view(-1,784)
        data = data.to(device) 
        target = target.float().to(device)

        output = model(data)
        optimizer.zero_grad()
        #loss = Loss(output,target)
        loss = F.mse_loss(output, target) #Virtually the same as nn.MSELoss
        loss.backward()
        optimizer.step()
        loss_cum.append(loss.item())
        Acc += torch.round(output.data.cpu()).squeeze(1).long().eq(target.data.cpu().long()).sum()
    print("")
    print("Loss: %0.3f"%(np.array(loss_cum).mean()))
    print("Acc: %0.2f"%(float(Acc*100)/len(train_loader.dataset)))

```

What does the loss value over the epochs tells you?, why the does the accuracy seems to reach a plateau at 0.2 accuracy? Do you think this a problem on the optimization parameters, feel free to change them.

## 2. A Bigger network

The result on our first network was rather lackluster, but remember it is the smallest network than can be designed, and is still better than a random classification, we are probably heading in the right direction and just need a bigger network to improve the results, let's modify the network to use 10 neurons on the first layer, followed by our current 1 neuron layer.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define a new neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10) 
        self.fc2 = nn.Linear(10, 1) #Add 1 fully connected layer with 10 neurons
        self.relu = nn.ReLU() #Is this strictly necesary?

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#display the model summary, check the number of parameters

model = Net()
model.to(device) 

print(model)
num_params=0
for p in model.parameters():
    num_params+=p.numel()
print("The number of parameters {}".format(num_params))

```

This time our classification accuracy is close to 0.4, a big jump that suggest to keep adding several more layers (convolutional and non linear filters) which will, likely, improve our results. Try adding several more layers to the current network, what's the best accuracy you can get?

## 3. A convolutional neural network

The very first step in the first example was to destroy the images spatial information in order to work with 1d neural layers. Convolutional neural layers allow us to keep this spatial information.

The overall setup for the experiment is the same, however, as we now have a much bigger network, we can properly formulate the problem as a classification problem rather than a regression one, hence we modify the shape of the labels by using 

Define the convolutional network as:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #layer with 64 2d convolutional filter of size 3x3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3) #Channels input: 1, c output: 63, filter of size 3
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc = nn.Linear(320, 10)    
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer

        x = F.max_pool2d(F.relu(self.conv2(x)), 2))

        x = F.Dropout(x, 0.25, training=self.training)#Try to control overfit on the network, by randomly excluding 25% of neurons on the last #layer during each iteration

        x = F.max_pool2d(F.relu(self.conv3(x)), 2))

        x = F.Dropout(x, 0.25, training=self.training)

        #Turn the 2d response of the network into a 1d array so we can output a 1x10 array
        x = x.view(-1, 320)
        x = self.fc(x)
        return x

#display the model summary, check the number of parameters

model = Net()
model.to(device)

print(model)
num_params=0
for p in model.parameters():
    num_params+=p.numel()
print("The number of parameters {}".format(num_params))

```

And use a suitable loss function for a classification problem:

```python
#As we have 10 different outputs, we need to change the cost function: Softmax+CrossEntropy
Loss = nn.CrossEntropyLoss()
```

## Final thoughts
While there are no deliverables for this lab, you might want to play around with this dataset and the convolutional neural layers.

# CHALLENGE
![](http://www.reactiongifs.com/r/2013/11/cute-fight.gif)

Although we do not have deliverables for this lab, we do have a small challenge :D. 
I will explain it in detail during our class. Just keep in mind that the winner(s) of this challenge will be able to **replace the worst grade** with a beautiful **5.0**. 
I will create a leader-board so you can track in real time the best accuracy and try to beat him/her.

