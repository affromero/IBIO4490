import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()

    def forward(self, x, dropout=0.5):
        return F.dropout(x, dropout, training=self.training)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.dropout=nn.Dropout()

    def forward(self, x):
        return self.dropout(x)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arange', type=int, default=20, choices=[5,10,15,20])
    parser.add_argument('--net', type=str, default='0', choices=['0','1'])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    if args.net=='0':
        model = Net0()
    else:
        model = Net1()

    model.to(device)

    if args.test: model.eval()
    else: model.train()
    
    x = torch.arange(args.arange).float().to(device)
    out = model(x, dropout=args.dropout)
    print(out.data.cpu().numpy().flatten())
    
    
    

