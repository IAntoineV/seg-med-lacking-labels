import torch
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from typing_extensions import override


class SegmentationDataset(Dataset):
    def __init__(self, images, labels, num_classes=55):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W).
            labels (np.ndarray): Array of flattened labels with shape (N, H*W).
            num_classes (int): Number of segmentation classes.
        """
        self.H = images.shape[1]
        self.W = images.shape[2]
        self.images = torch.from_numpy(images)
        n,_ = labels.shape
        self.labels = torch.from_numpy(labels.reshape(n,self.H, self.W))
        self.num_classes = num_classes


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image =  image.unsqueeze(0) / 255.0  # Normalize to [0,1]
        label = self.labels[idx]

        return image, label

    #def __getitems__(self, indices):
    #    image = self.images[indices].unsqueeze(1)
    #    image = 2* image / 255.0 -1  # Normalize to [-1,1]
    #    label = self.labels[indices]
    #    return image, label


class SegmentationDatasetRGB(SegmentationDataset):

    @override
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return torch.tile(img, (1,self.H, self.W)), label



def load_img_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)

def train_labels():
    labels_train = pd.read_csv("y_train.csv", index_col=0).T
    return labels_train.to_numpy()

def get_data(test_size=0.2,device = "cpu"):
    data_dir = Path("./")
    x = load_img_dataset(data_dir / "train-images")
    y =train_labels()

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42)
    dataset_train = SegmentationDataset(X_train, y_train)
    dataset_test = SegmentationDataset(X_test, y_test)
    return dataset_train, dataset_test






class SegmentationDatasetDisk(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_files = sorted(self.image_dir.glob("*.png"), key=lambda x: int(x.stem))
        self.labels = pd.read_csv(label_csv, index_col=0).T.to_numpy()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        # Convert grayscale to PIL image
        image = Image.fromarray(image, mode="L")  # "L" mode is for grayscale

        # Convert grayscale to RGB (3 channels)
        image = image.convert("RGB")  # Converts (H, W) → (H, W, 3)
        h,w = image.size
        if self.transform:
            image = self.transform(image)


        return image , torch.tensor(self.labels[idx], dtype=torch.float32).reshape(h, w)


def get_dataloader_disk(image_dir, label_csv, batch_size=32, test_size=0.2, num_workers=4, transform=transforms.ToTensor()):
    dataset = SegmentationDatasetDisk(image_dir, label_csv, transform=transform)
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# Example usage:
# train_loader, test_loader = get_dataloader("./train-images", "y_train.csv")


