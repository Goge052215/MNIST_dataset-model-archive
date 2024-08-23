import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adam import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)  # New conv layer
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(3136, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)  # New conv layer
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(train_loader_tqdm):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item(), refresh=False)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMNISTModel().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, epochs=20, steps_per_epoch=len(train_loader))

num_epochs = 20

print("Training the model...")
for epoch in range(1, num_epochs + 1):
    avg_loss = train(model, device, train_loader, optimizer, criterion, epoch)
    accuracy = test(model, device, test_loader, criterion)
    scheduler.step()

final_accuracy = test(model, device, test_loader, criterion)
print(f"Final Test Accuracy: {final_accuracy:.2f}%")