import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size=1000

#transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform_train = transforms.Compose([transforms.ToTensor()])
data_train = datasets.MNIST('data', train=True, transform = transform_train)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

data_test = datasets.MNIST('data', train=True, transform = transform_train)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

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
        self.fc = nn.Linear(784, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

model = Net()
model.to(device)
print_network(model, 'fc 1 layer')

print(model.state_dict().keys())
print(model.state_dict()['fc.weight'].shape)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

Loss = nn.MSELoss()

def train(epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch: {}".format(epoch)):
        data = data.view(-1,784)
        data = data.to(device)
        target = target.float().to(device)

        output = model(data)
        optimizer.zero_grad()
        loss = Loss(output,target)
        #loss = F.mse_loss(output, target) #Practically the same
        loss.backward()
        optimizer.step()
        loss_cum.append(loss.item())
        Acc += torch.round(output.data.cpu()).squeeze(1).long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f"%(np.array(loss_cum).mean()))
    print("Acc: %0.2f"%(float(Acc*100)/len(train_loader.dataset)))

epochs=20
for epoch in range(epochs):
    train(epoch)

