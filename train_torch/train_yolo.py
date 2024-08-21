import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from yolov5 import YOLOv5
from torch.optim.rmsprop import RMSprop

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Load YOLOv5 model
class YOLOv5Wrapper(torch.nn.Module):
    def __init__(self, yolov5_model):
        super(YOLOv5Wrapper, self).__init__()
        self.yolov5_model = yolov5_model

    def forward(self, x):
        return self.yolov5_model(x)

yolov5_model = YOLOv5('yolov5s.pt')  # Use a pre-trained YOLOv5 model
model = YOLOv5Wrapper(yolov5_model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=0.001)


def train(model, train_loader, criterion, optimizer, num_epochs=18):
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
    torch.save(model.state_dict(), '../models/yolo_mnist_model.pth')  # Save the trained model


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


if __name__ == '__main__':
    train(model, train_loader, criterion, optimizer)
    evaluate(model, test_loader)
