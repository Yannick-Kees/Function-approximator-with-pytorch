import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from seaborn import scatterplot

# create random points
Train = torch.rand(400,2)
Test = torch.rand(100,2)

#create labels
f = lambda x: np.sin(10*x)
decider = lambda x: 0.0 if f(x[0])>x[1] else 1.0
target = [decider(x) for x in Train]

#Plot training data
plt.figure()
scatterplot(np.array(Train).T[0],np.array(Train).T[1],  hue=target)
h = np.arange(0,1,.02)
plt.plot(h,f(h), color='r')
plt.ylim(0,1)
plt.title("Training Data Set")
 

class MyNetwork(nn.Module):
    # Neuronal Network
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.lin1 = nn.Linear(2,16)
        self.lin2 = nn.Linear(16,32)
        self.lin3 = nn.Linear(32,128)
        self.lin5 = nn.Linear(128,1)

    
    def forward(self, x):
        # Feed forward function
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin5(x)
        return x

netz = MyNetwork()

for _ in range(5000):
    # training the network
    # feed forward
    input = Variable(Train)
    out = netz(input)
    #Compute loss
    target_nn = Variable(torch.tensor( [[t] for t in target] ))
    criterion = nn.MSELoss()
    loss = criterion(out, target_nn)
    #backpropagation
    netz.zero_grad()
    loss.backward()
    optimizer = optim.SGD(netz.parameters(), lr=.05)
    optimizer.step()
    
cut = lambda x: 1 if x>0.5 else 0
training_output_scores = np.ravel(netz(Variable(Test)).detach().numpy())

#plot test data
plt.figure()
scatterplot(np.array(Test).T[0],np.array(Test).T[1],  hue= [cut(x) for x in training_output_scores] )
plt.plot(h,f(h), color='r')
plt.ylim(0,1)
plt.title("Test Data Set")
plt.show()