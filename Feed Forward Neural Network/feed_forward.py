import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784 # image size is 28x28
hidden_size = 500 # can be tuned for performanance

num_classes = 10 # 0-9 digits, 10 classes
n_epochs = 2 
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size,
                                          shuffle=False) # shuffling doesn't matter for the evaluation

# Uncomment to test
# examples = iter(train_loader)
# print(len(examples)) # 600 in total, each batch will load 100 
# samples, labels = examples.next()
# # sample size = (100, 1, 28, 28), labels size = (100)
# print(samples.shape, labels.shape) 

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes): # output size = num_classes
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        
        # Define linear layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() 
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    # Define forward pass
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        # No activation and softmax at the end
        return x
        
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images
        # # Initial size = (100, 1, 28, 28) -> Input size = (100, 784)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
         
        # Forward pass
        outputs = model(images)
        
        # Loss
        loss = criterion(outputs, labels)
        
        # Clear the gradients of the model parameters
        optimizer.zero_grad()
        
        # Backward propagation and optimize
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0: # Print every 100th step
            print(f'Epoch: [{epoch+1}/{n_epochs}], Step: [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test 
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # Reshape images
        # Initial size = (100, 1, 28, 28) -> Input size = (100, 784)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
         
        outputs = model(images)
        
        # torch.max returns (value, index)
        _, predictions = torch.max(outputs, dim=1)
        n_samples += labels.size(0) 
        n_correct += (predictions == labels).sum().item()
        
    acc = (n_correct / n_samples) * 100.0
    print(f'Accuracy of the network on the 10000 test images: {acc} %')