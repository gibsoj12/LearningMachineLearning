import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#training dataset
train = datasets.MNIST("", train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

#testing dataset (don't want to overfit)
test = datasets.MNIST("", train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64) #for every layer we will have a flattened 28*28 image, output will be #neurons of the layer
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) #This is 10 because there are 10 possible digits


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.softmax(x, dim=1)

net = Net()
optimizer = optim.Adam(net.parameters(), lr = 1e-3)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        #data is a batch of feature sets and labels
        X, y = data
        net.zero_grad() #Zero gradiant after every batch
        output = net(X.view(10,28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(10,28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy:", round(correct/total, 3))