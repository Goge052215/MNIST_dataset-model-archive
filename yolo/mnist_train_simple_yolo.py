# Import reqiured modules
import torch
import torch.nn as nn
import torch.optim.adamw as optim
from torch.optim.rmsprop import RMSprop  # Optimizer added
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars
from torch.optim.lr_scheduler import StepLR


# Based on YOLO's mechanism we build a conv network
class SimpleYOLO(nn.Module):
    def __init__(self):
        super(SimpleYOLO, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32) # Batch the data
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4 + 10)
        self.dropout = nn.Dropout(0.5)  # Dropping off the nodes
        self.leaky_relu = nn.LeakyReLU(0.1)  # ReLU
        self.softmax = nn.Softmax(dim=1)  # Add softmax layer

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x[:, 4:] = self.softmax(x[:, 4:])  # Apply softmax to classification output
        return x


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # Ensure fast so no action is needed
])

# Fetch the dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
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

        train_loader_tqdm.set_postfix(loss=loss.item(), refresh=False)


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
model = SimpleYOLO().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # In this scenario RMS is better than Adam
criterion = nn.MSELoss()
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

final_accuracy = 0  # To store the final accuracy after all epochs

num_epochs = 20

for epoch in range(1, num_epochs + 1):
    avg_loss = train(model, device, train_loader, optimizer, criterion, epoch)
    accuracy = test(model, device, test_loader, criterion)
    scheduler.step(avg_loss)  # Update learning rate based on training loss

    torch.save(model.state_dict(), 'models/mnist_simple_yolo.pth')
    final_accuracy = accuracy

print(f"Accuracy: {final_accuracy:.2f}%")
