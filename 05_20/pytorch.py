# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt, numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.unsqueeze(torch.linspace(-2,2,100), dim = 1)
y = 1 + torch.sin(np.pi/4.0*x)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = torch.sigmoid(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=2, n_output=1)
net = net.cuda()
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.3)
loss_func = torch.nn.MSELoss()

for t in range(100):
    x = x.cuda()
    y = y.cuda()
    
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    x = x.cpu()
    y = y.cpu()
    prediction = prediction.cpu()
    loss = loss.cpu()
    
    if t % 5 == 0:
        plt.cla()
        plt.plot(x.data.numpy(),y.data.numpy(),'b',lw=1)
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-.',lw=1)
        plt.text(0,0,'Loss=%.4f' % loss.data.numpy(),fontdict={'size' :20,'color': 'red'})
        plt.pause(0.1)


