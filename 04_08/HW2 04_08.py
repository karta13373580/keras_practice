import torch
import torch.nn.functional as F
import numpy as np

x = torch.unsqueeze(torch.linspace(-2,2,100),dim = 1)
y = 1 + torch.sin(np.pi/4.0*x)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
        
    def forward(self,x):
        x = F.sigmoid(self.hidden(x))
        x = self.predict(x)
        return x
    
net = Net(n_feature=1, n_hidden=2, n_output=1)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.3)
loss_func = torch.nn.MSELoss()

