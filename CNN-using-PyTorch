# -*- coding: utf-8 -*-
"""
Here, we build a simple convolutional neural network in PyTorch and train it to recognize handwritten digits using the MNIST dataset.

MNIST contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels, and centered to reduce preprocessing and get started quicker.
"""

from google.colab import drive
drive.mount ('/content/drive')

#Setting up the Pytorch enviornment
import torch
import torchvision #The torchvision package consists of popular datasets, model architectures, and common image transformations for computer visio

# Defining the Hyperparameters:
# number of epochs defines how many times we'll loop over the complete training dataset
# learning_rate and momentum are hyperparameters for the optimizer
# repeatable experiments we have to set random seeds
#cuDNN uses nondeterministic algorithms with: torch.backends.cudnn.enabled = False
#If CuDNN will use deterministic algorithms for these operations, it will always give same input and parameters, and hence yeilds the same output.
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#load the MNIST dataset from the DataLoaders with TorchVision.
#TorchVision offers a lot of handy transformations, such as cropping or normalization
#A batch_size of 64 for training and size 1000 for testing on this dataset.
#The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation of the MNIST dataset.
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader) #The enumerate() function adds counter to an iterable and returns it. The returned object is an enumerate object.

batch_idx, (example_data, example_targets) = next(examples)

print(type(examples))

# converting to list
print(list(examples)[0:1])

#one test data batch consists of the following dimesions
#we have 1000 examples of 28x28 pixels in grayscale
example_data.shape

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

"""**Buiding the Neural Network**

We build a CNN model with two 2-D convolutional layers followed by two fully-connected (or linear) layers. As activation function we'll choose rectified linear units (ReLUs in short) and as a means of regularization we'll use two dropout layers.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)#-1 implies dynamic number of rows to be included when reshaping
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,1)

#initialize the network and the optimizer.
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

"""**Training the Model**"""

#to create a nice training curve later on we also create two lists for saving training and testing losses.
# On the x-axis we want to display the number of training examples the network has seen during training.
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

#Defining the train()
#Here we manually set the gradients to zero using optimizer.zero_grad() since PyTorch by default accumulates gradients.
#We then produce the output of our network (forward pass) and compute a negative log-likelihodd loss between the output and the ground truth label
#The backward() call we now collect a new set of gradients which we propagate back into each of the network's parameters using optimizer.step().
#Neural network modules as well as optimizers have the ability to save and load their internal state using .state_dict()
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)#nll_loss() is the negative log likelihood loss
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), '/content/drive/MyDrive/ML Nov/model.pth')
      #Neural network modules as well as optimizers can continue training from previously saved state dicts if needed - we'd just need to call .load_state_dict(state_dict).
      torch.save(optimizer.state_dict(), '/content/drive/MyDrive/ML Nov/optimizer.pth')

#network.state_dict()

##Defining the test()
#Here we sum up the test loss and keep track of correctly classified digits to compute the accuracy of the network.
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

"""Using the context manager no_grad() we can avoid storing the computations done producing the output of our network in the computation graph.

Time to run the training! We'll manually add a test() call before we loop over n_epochs to evaluate our model with randomly initialized parameters.
"""

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

"""With just 3 epochs of training we already managed to achieve 97% accuracy on the test set!"""

print(train_losses)

print(test_losses)
print(test_counter)

"""**Evaluating the Model Performance**"""

#plot our training curve.
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses[0:4], color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig

#look at a few examples as we did earlier and compare the model's output.
with torch.no_grad():
  output = network(example_data)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
fig

"""**Continued Training from Checkpoints**

Let's continue training the network, or rather see how we can continue training from the state_dicts we saved during our first training run. We'll initialize a new set of network and optimizers.
"""

continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)

#Using .load_state_dict() we can now load the internal state of the network and optimizer when we last saved them.
network_state_dict = torch.load('/content/drive/MyDrive/ML Nov/model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load('/content/drive/MyDrive/ML Nov/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

for i in range(4,9):
  test_counter.append(i*len(train_loader.dataset))
  train(i)
  test()

