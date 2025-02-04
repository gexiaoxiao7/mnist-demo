import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import platform
import os
from model import ConvNet
from torch.optim import lr_scheduler

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


ConvModel = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(ConvModel.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
epoch = 2

def train(num_epochs, _model, _device, _train_loader, _optimizer, _lr_scheduler):
    _model.train()
    _lr_scheduler.step()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)
            output = _model(samples.reshape(-1, 1, 28, 28))
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch + 1, num_epochs, i + 1, loss.item()))


def test(_test_loader, _model, _device):
    _model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = ConvModel(data.reshape(-1, 1, 28, 28))
            loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    loss /= len(_test_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))
    print('\nTest Accuracy: ({:.3f}%) '.format(100. * correct / len(_test_loader.dataset)))

for epoch in range(1, epoch + 1):
    train(epoch, ConvModel, device, train_loader , optimizer, exp_lr_scheduler)
    test(test_loader, ConvModel, device)
    test(train_loader, ConvModel, device)

# save model
if not os.path.exists("output"):
    os.makedirs("output")
torch.save(ConvModel.state_dict(),"output/ConvModel.pth")