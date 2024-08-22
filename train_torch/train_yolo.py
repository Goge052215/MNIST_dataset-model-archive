import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.rmsprop import RMSprop  # Import RMSprop from torch.optim.rmsprop
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars


class SimpleYOLO(nn.Module):
    def __init__(self):
        super(SimpleYOLO, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 1 input channel (grayscale), 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*22*22, 128)  # Adjusted to match the output size of conv3
        self.fc2 = nn.Linear(128, 4 + 10)  # (x, y, w, h) + 10 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Ensure the batch size dimension is preserved
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

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
optimizer = RMSprop(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

final_accuracy = 0  # To store the final accuracy after all epochs

num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    accuracy = test(model, device, test_loader, criterion)
    final_accuracy = accuracy

print(f"Final Accuracy after all epochs: {final_accuracy:.2f}%")
