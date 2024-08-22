from ultralytics import YOLO
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

# Download the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Directory for YOLOv8 dataset
os.makedirs('mnist_yolo/train/images', exist_ok=True)
os.makedirs('mnist_yolo/train/labels', exist_ok=True)
os.makedirs('mnist_yolo/val/images', exist_ok=True)
os.makedirs('mnist_yolo/val/labels', exist_ok=True)


def create_yolo_labels(data_loader, split):
    for idx, (data, target) in enumerate(data_loader):
        # Convert image to NumPy array and save it
        image = data.squeeze().numpy() * 255
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_path = f'mnist_yolo/{split}/images/{idx}.jpg'
        cv2.imwrite(img_path, image)

        # Create label file
        label_path = f'mnist_yolo/{split}/labels/{idx}.txt'
        h, w = image.shape[:2]
        bbox = [0.5, 0.5, 1.0, 1.0]  # x_center, y_center, width, height normalized
        label = f"{target.item()} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
        with open(label_path, 'w') as f:
            f.write(label)


create_yolo_labels(train_loader, 'train')
create_yolo_labels(test_loader, 'val')

model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model (nano version)
results = model.train(data='mnist_yolo.yaml', epochs=10)

# Evaluate the model on the test dataset
metrics = model.val()

print(f"Model Performance: {metrics}")
