from ultralytics import YOLO
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

base_dir = os.path.expanduser('~/MNIST_yolo')  # Base directory within user's home directory
os.makedirs(base_dir, exist_ok=True)

# Use torchvision to download MNIST
train_dataset = datasets.MNIST(root=os.path.join(base_dir, 'data'), train=True, download=True)
test_dataset = datasets.MNIST(root=os.path.join(base_dir, 'data'), train=False, download=True)

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use the transform when creating the datasets
train_dataset = datasets.MNIST(root=os.path.join(base_dir, 'data'), train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=os.path.join(base_dir, 'data'), train=False, download=True, transform=transform)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Update the create_yolo_labels function
def create_yolo_labels(data_loader, split):
    for idx, (data, target) in enumerate(data_loader):
        # Convert image to NumPy array and save it
        image = data.squeeze().numpy() * 255
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_path = os.path.join(base_dir, f'mnist_yolo/{split}/images/{idx}.jpg')
        cv2.imwrite(img_path, image)

        # Create label file
        label_path = os.path.join(base_dir, f'mnist_yolo/{split}/labels/{idx}.txt')
        h, w = image.shape[:2]
        bbox = [0.5, 0.5, 1.0, 1.0]  # x_center, y_center, width, height normalized
        label = f"{target.item()} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
        with open(label_path, 'w') as f:
            f.write(label)

# Call the function
create_yolo_labels(train_loader, 'train')
create_yolo_labels(test_loader, 'val')

model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model (nano version)
results = model.train(data=os.path.join(base_dir, 'mnist_yolo.yaml'), epochs=10)

metrics = model.val()

print(f"Model Performance: {metrics}")
