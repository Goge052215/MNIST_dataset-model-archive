# This plan is temporarily abandoned

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

# Define transformations
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, shear=15, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 2500)
        self.bn1 = nn.BatchNorm1d(2500)
        self.fc2 = nn.Linear(2500, 2000)
        self.bn2 = nn.BatchNorm1d(2000)
        self.fc3 = nn.Linear(2000, 1500)
        self.bn3 = nn.BatchNorm1d(1500)
        self.fc4 = nn.Linear(1500, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc5 = nn.Linear(1000, 500)
        self.bn5 = nn.BatchNorm1d(500)
        self.fc6 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.6)  # Increased dropout rate

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc6(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = DeepNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.adam.Adam(model.parameters(), lr=0.001)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/(i+1))
        scheduler.step(running_loss)


train(model, train_loader, criterion, optimizer, scheduler)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')


evaluate(model, test_loader)
