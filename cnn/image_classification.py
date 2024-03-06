import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Define a CNN model.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers.
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=16, 
                               kernel_size=5, 
                               stride=1, 
                               padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, 
                               out_channels=32, 
                               kernel_size=5, 
                               stride=1, 
                               padding=2)
        
        # Max pooling layer.
        self.pool = nn.MaxPool2d(kernel_size=2, 
                                 stride=2, 
                                 padding=0)
        
        # Fully connected layers.
        self.fc1 = nn.Linear(32 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        """
        Forward pass function for the CNN model.

        Args:
            x (torch.Tensor):   Input tensor of shape (batch_size, channels,
                                height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Convolutional layers followed by max pooling.
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten the output for fully connected layers.
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation.
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Data preprocessing and augmentation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                             train=True, 
                                             download=True, 
                                             transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)

# Data loaders.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=4, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=4, 
                                          shuffle=False)

# Initialize the CNN model.
model = CNN()

# Define loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model.
for epoch in range(2):  # Loop over the dataset multiple times.
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        # Get the inputs; data is a list of [inputs, labels].
        inputs, labels = data

        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward + backward + optimize.
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics.
        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches.
            print('[%d, %5d] loss: %.3f' % (epoch + 1, 
                                            i + 1, 
                                            running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Evaluate the model.
correct = 0
total = 0

#### TODO ####
"""
Task: Implement model evaluation on the test dataset.

Steps to Implement:
1. Iterate over the test_loader.
2. Extract images and labels from each batch.
3. Pass images through the trained model to get outputs.
4. Get predicted labels using torch.max.
5. Compare predicted labels with ground truth labels.
6. Calculate the number of correct predictions and update the total.
7. Compute the accuracy using correct predictions and total images.
"""

print('Accuracy of the network on the 10000 test images: %d %%' % 
      (100 * correct / total))
