import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 28
hidden_size = 128
sequence_length = 28
num_classes = 10 # digits from 0-9
num_layers = 2

num_epochs = 2
batch_size = 100
learning_rate = 0.001


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Use built-in PyTorch RNN model
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # batch_first=True means need batch_size as first dimension
        # input shape needs to be -> (batch_size, sequence_length, input_size) -> (N, 28, 128)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    
    def forward(self, x):
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        out, _ = self.rnn(x, h0)
        # out shape -> (batch_size, sequence_length, hidden_size) -> (N, 28, 128)
        out = out[:, -1, :] # only want the last timestep -> (N, 128)
        out = self.fc(out)
        return out
    
    
rnn = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)     

# Train the model
n_total_steps = len(train_loader)  
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Original shape -> (100, 1, 28, 28)
        # Reshape to input shape to -> (batch_size, sequence_length, input_size) -> (100, 28, 28)
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = rnn(images)
        
        # Loss
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        
        # Backward pass and update
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step: {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')
            
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # Original shape -> (100, 1, 28, 28)
        # Reshape to input shape to -> (batch_size, sequence_length, input_size) -> (100, 28, 28)
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = rnn(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs.data, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
    acc = (n_correct / n_samples) * 100
    print(f'Accuracy of the network on 10,000 test images: {acc:.2f} %')