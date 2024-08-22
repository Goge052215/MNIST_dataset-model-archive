from ultralytics import YOLO
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import requests
import gzip
import shutil

base_dir = os.path.expanduser('~/MNIST_yolo')  # Base directory within user's home directory
os.makedirs(base_dir, exist_ok=True)

# URLs for MNIST dataset
mnist_urls = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
}


def download_and_extract(url, output_path):
    gz_path = output_path + ".gz"
    with requests.get(url, stream=True) as r:
        with open(gz_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)  # Remove the gz file after extraction


for key, url in mnist_urls.items():
    output_path = os.path.join(base_dir, "data", os.path.basename(url).replace(".gz", ""))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    download_and_extract(url, output_path)

# Prepare the dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def extract_mnist_images(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'rb') as f:
        _ = f.read(16)  # Skip the header
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

        for idx, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_path = os.path.join(output_dir, f"{idx}.jpg")
            cv2.imwrite(img_path, img)


extract_mnist_images(os.path.join(base_dir, "data/t10k-images-idx3-ubyte"),
                     os.path.join(base_dir, "mnist_yolo/val/images"))


def create_labels(image_dir, label_dir, labels):
    os.makedirs(label_dir, exist_ok=True)

    for idx, label in enumerate(labels):
        label_path = os.path.join(label_dir, f"{idx}.txt")
        bbox = [0.5, 0.5, 1.0, 1.0]  # x_center, y_center, width, height normalized
        with open(label_path, 'w') as f:
            f.write(f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")


labels = np.frombuffer(open(os.path.join(base_dir, "data/t10k-labels-idx1-ubyte"), 'rb').read(), dtype=np.uint8, offset=8)
create_labels(os.path.join(base_dir, "mnist_yolo/val/images"),
              os.path.join(base_dir, "mnist_yolo/val/labels"), labels)

train_dataset = datasets.MNIST(root=os.path.join(base_dir, 'data'), train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=os.path.join(base_dir, 'data'), train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Directory for YOLOv8 dataset
os.makedirs(os.path.join(base_dir, 'mnist_yolo/train/images'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'mnist_yolo/train/labels'), exist_ok=True)


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


create_yolo_labels(train_loader, 'train')
create_yolo_labels(test_loader, 'val')

model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model (nano version)
results = model.train(data=os.path.join(base_dir, 'mnist_yolo.yaml'), epochs=10)

metrics = model.val()

print(f"Model Performance: {metrics}")
