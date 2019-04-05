
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = torch.from_numpy(np.array(x_train))
    y_train = torch.from_numpy(np.array(y_train)).type(torch.FloatTensor)
    x_test = torch.from_numpy(np.array(x_test))
    y_test = torch.from_numpy(np.array(y_test)).type(torch.FloatTensor)

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.view(x_train.size(0), 48, 48)
    x_test = x_test.view(x_test.size(0), 48, 48)
    y_train = y_train.view(y_train.size(0), 1)
    y_test = y_test.view(y_test.size(0), 1)

    print(x_train.size(0), 'train samples')
    print(x_test.size(0), 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(48*48, 1, bias=True) #Add 1 fully connected layer with 1 neuron

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def compute_loss(pred, gt):
    J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
    return J

def sigmoid(x):
    return 1/(1+np.exp(-x))

def train(model):
    x_train, y_train, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 40000 # Change if you want
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    for i in range(epochs):
        LOSS = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            optimizer.zero_grad()
            out = model(_x_train) 
            loss = criterion(out, _y_train)
            # compute_loss(out.data.cpu().numpy(), _y_train.data.cpu().numpy())
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
        out = model(x_test)      
        loss_test = criterion(out, y_test).item()
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, torch.FloatTensor(LOSS).mean(), loss_test))

if __name__ == '__main__':
    model = Net()
    model.to(device) 
    train(model)

