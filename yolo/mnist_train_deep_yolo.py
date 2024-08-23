import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adam import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Create 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

class ImprovedYOLO(nn.Module):
    def __init__(self):
        super(ImprovedYOLO, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4 + 10)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(self.leaky_relu(self.batch_norm4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Split the training data into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader_tqdm):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        batch_size = data.shape[0]
        target_bbox = torch.tensor([[0.5, 0.5, 1.0, 1.0]] * batch_size).to(device)
        target_onehot = torch.eye(10, device=device)[target]  # Ensure target_onehot is on the same device
        target_combined = torch.cat((target_bbox, target_onehot), dim=1)

        output = model(data)
        loss = criterion(output, target_combined)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item(), refresh=False)
    avg_loss = total_loss / len(train_loader)
    return avg_loss  # Return average loss for the epoch


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            batch_size = data.shape[0]
            target_bbox = torch.tensor([[0.5, 0.5, 1.0, 1.0]] * batch_size).to(device)
            target_onehot = torch.eye(10, device=device)[target]  # Ensure target_onehot is on the same device
            target_combined = torch.cat((target_bbox, target_onehot), dim=1)

            test_loss += criterion(output, target_combined).item()
            pred = output[:, 4:].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedYOLO().to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose='True')

# Use the existing loaders
train_loader = train_loader  # Already defined
valid_loader = val_loader  # Rename val_loader to valid_loader for consistency

num_epochs = 50
patience = 10
best_valid_loss = float('inf')
counter = 0

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
    valid_accuracy = test(model, device, valid_loader, criterion)
    test_accuracy = test(model, device, test_loader, criterion)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    
    scheduler.step(valid_accuracy)
    
    if valid_accuracy < best_valid_loss:
        best_valid_loss = valid_accuracy
        counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('models/best_model.pth'))
final_accuracy = test(model, device, test_loader, criterion)
print(f"Final Test Accuracy: {final_accuracy:.2f}%")