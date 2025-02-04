import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import platform
import os
from model import Net


if platform.system() == "Darwin": # use gpu on mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else: # use gpu on linux / windows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 100
transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.1307),(0.3081))
])

# Data set
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)


net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum = 0.9)

# train
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(train_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = F.softmax(net(images))
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss}, Accuracy: {100 * correct / len(train_dataset)}')

# save model
if 'output' not in os.listdir():
    os.mkdir('output')
torch.save(net.state_dict(), "output/mnist_cnn.pt")