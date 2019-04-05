import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #layer with 64 2d convolutional filter of size 3x3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3) #Channels input: 1, c output: 63, filter of size 3
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc = nn.Linear(32, 10)    
    
    def forward(self, x, verbose=False):
        if verbose: "Output Layer by layer"
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if verbose: print(x.size())
        x = F.dropout(x, 0.25, training=self.training)#Try to control overfit on the network, by randomly excluding 25% of neurons on the last #layer during each iteration
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        if verbose: print(x.size())
        x = F.dropout(x, 0.25, training=self.training)
        if verbose: print(x.size())
        #ipdb.set_trace()
        x = x.view(-1, 32)
        if verbose: print(x.size())
        x = self.fc(x)
        if verbose: print(x.size())
        return x

    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.CrossEntropyLoss()
        
def get_data(batch_size):
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_train = transforms.Compose([transforms.ToTensor()])
    data_train = datasets.MNIST('data', train=True, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_test = datasets.MNIST('data', train=False, transform = transform_train)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)   
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))

def test(data_loader, model, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)
        target = target.to(device).requires_grad_(False)

        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss Test: %0.3f | Acc Test: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
    
if __name__=='__main__':
    epochs=20
    batch_size=1000
    TEST=False
    train_loader, test_loader = get_data(batch_size)

    model = Net()
    model.to(device)

    model.training_params()
    print_network(model, 'Conv network')    

    #Exploring model
    data, _ = next(iter(train_loader))
    _ = model(data.to(device).requires_grad_(False), verbose=True)

    for epoch in range(epochs):
        train(train_loader, model, epoch)
        if TEST: test(test_loader, model, epoch)



