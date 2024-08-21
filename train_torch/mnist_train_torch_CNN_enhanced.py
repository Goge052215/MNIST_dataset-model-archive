import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import GaussianBlur
from torch.optim.rmsprop import RMSprop

def elastic_transform(image, alpha, sigma):
    shape = image.shape[1:]
    dx = torch.randn(shape) * sigma
    dy = torch.randn(shape) * sigma
    blur_transform = GaussianBlur(kernel_size=15, sigma=sigma)
    dx = blur_transform(dx.unsqueeze(0)).squeeze(0)
    dy = blur_transform(dy.unsqueeze(0)).squeeze(0)

    x, y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
    x, y = x.float(), y.float()
    x_new = x + dx * alpha
    y_new = y + dy * alpha

    grid = torch.stack([x_new, y_new], dim=-1).unsqueeze(0)
    return F.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='reflection').squeeze(0)


# Define transformations with elastic distortions
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: elastic_transform(x, alpha=34, sigma=4)),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class EnhancedCNN_V2(nn.Module):
    def __init__(self):
        super(EnhancedCNN_V2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(40 * 4 * 4, 150)
        self.fc2 = nn.Linear(150, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 40 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create an ensemble of 7 CNN models
committee = [EnhancedCNN_V2().to(device) for _ in range(7)]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sum([list(model.parameters()) for model in committee], []), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


def train(committee, train_loader, criterion, optimizer, scheduler, num_epochs=20):
    for model in committee:
        model.train()
        
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = sum(model(images) for model in committee) / len(committee)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/(i+1))
        
        scheduler.step(running_loss / len(train_loader))
    
    for idx, model in enumerate(committee):
        torch.save(model.state_dict(), f'models/cnn_deep_model_{idx}.pth')  # Save each model in the committee


def evaluate(committee, test_loader):
    for model in committee:
        model.eval()
        
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = sum(model(images) for model in committee) / len(committee)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    train(committee, train_loader, criterion, optimizer, scheduler)
    evaluate(committee, test_loader)
