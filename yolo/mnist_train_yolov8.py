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
    # Create full directory structure
    images_dir = os.path.join(base_dir, 'mnist_yolo', split, 'images')
    labels_dir = os.path.join(base_dir, 'mnist_yolo', split, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for idx, (data, target) in enumerate(data_loader):
        # Convert image to NumPy array and save it
        image = data.squeeze().numpy() * 255
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_path = os.path.join(images_dir, f'{idx}.jpg')
        cv2.imwrite(img_path, image)

        # Create label file
        label_path = os.path.join(labels_dir, f'{idx}.txt')
        h, w = image.shape[:2]
        bbox = [0.5, 0.5, 1.0, 1.0]  # x_center, y_center, width, height normalized
        label = f"{target.item()} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
        with open(label_path, 'w') as f:
            f.write(label)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # Call the function
    create_yolo_labels(train_loader, 'train')
    create_yolo_labels(test_loader, 'val')

    yaml_content = f"""
    path: {base_dir}/mnist_yolo  # dataset root dir
    train: train/images  # train images (relative to 'path')
    val: val/images  # val images (relative to 'path')

    nc: 10  # number of classes
    names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # class names
    """

    yaml_path = os.path.join(base_dir, 'mnist_yolo.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Then use this path in the train() function
    model = YOLO('yolov8n.pt').to('cuda')
    results = model.train(
        data=yaml_path,
        epochs=15,
        imgsz=640,
        batch=64,
        device=0,
        amp=True,
        workers=4,
        augment=True,
        lr0=0.001,
        lrf=0.01,
        mosaic=1.0
    )

    metrics = model.val()

    print(f"Model Performance: {metrics}")
