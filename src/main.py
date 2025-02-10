import torch
import torch.nn as nn
import torch.optim as optim
import hydra

from tqdm import tqdm
from omegaconf import DictConfig
from dataset import get_data
from torch.utils.data import DataLoader
from torchvision import models
from utils import to_one_hot
from metrics import dice
import importlib

num_classes = 55
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="./config", config_name="models")
def main(cfg: DictConfig):
    data_train, data_test = get_data(test_size=0.05)

    dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=2)
    data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=2)

    module_name, class_name = cfg.model.class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class(pretrained=cfg.model.pretrained, num_classes=num_classes).to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_trainable_params:,}")

    def dice_criterion(outputs, labels):
        labels_hot = to_one_hot(labels, num_classes).to(device)
        dice_score = dice(labels_hot, torch.nn.functional.softmax(outputs['out'], dim=1))
        present_labels = torch.unique(labels)
        mask = torch.zeros(num_classes, device=device)
        mask[present_labels] = 1
        dice_masked = (mask * dice_score).sum() / len(present_labels)
        return 1 - dice_masked

    criterion = dice_criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 30

    print("Starting training loop...")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        it = 0
        for images, labels in dataloader:
            images = images.to(device)  # (B,1 , H, W)
            labels = labels.to(device)  # (B, H, W) with class indices
            optimizer.zero_grad()
            outputs = model(images)  # (B, num_classes, H, W)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            it += 1

        epoch_loss = running_loss / it
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        with torch.no_grad():
            dice_cum = 0
            for x_test, y_test in data_test_loader:
                output = model(x_test.to(device))
                dice_score = dice_criterion(output, y_test.to(device))
                dice_cum += (1 - dice_score.item()) * x_test.size(0)
            print("valid dice ", dice_cum / len(data_test))
    print("Training complete.")

if __name__ == '__main__':
    main()