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
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.MSELoss()
        
def get_data(batch_size):
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_train = transforms.Compose([transforms.ToTensor()])
    data_train = datasets.MNIST('data', train=True, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    return train_loader


def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Epoch: {}".format(epoch)):
        data = data.view(-1,784)
        data = data.to(device)
        target = target.float().to(device)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)
        #loss = F.mse_loss(output, target) #Practically the same
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        Acc += torch.round(output.data.cpu()).squeeze(1).long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f"%(np.array(loss_cum).mean()))
    print("Acc: %0.2f"%(float(Acc*100)/len(data_loader.dataset)))
    
if __name__=='__main__':
    epochs=20
    batch_size=1000
    train_loader = get_data(batch_size)

    model = Net()
    model.to(device)
    model.training_params()
    print_network(model, 'fc 2 layer non-linearity')    

    for epoch in range(epochs):
        train(train_loader, model, epoch)



