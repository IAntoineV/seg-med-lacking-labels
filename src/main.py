import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from dataset import get_data
from utils import to_one_hot
from metrics import dice
import importlib
from sam import YOLOv8SAM

num_classes = 55
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_configs = {
    "deeplabv3_mobilenet_v3_large": "torchvision.models.segmentation.deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50": "torchvision.models.segmentation.deeplabv3_resnet50",
    "sam_transformer": "sam.YOLOv8SAM",
}

def main(model_name="deeplabv3_mobilenet_v3_large"):
    data_train, data_test = get_data(test_size=0.05, rgb=True)

    dataloader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=2)

    cfg = model_configs[model_name]
    module_name, class_name = cfg.rsplit(".", 1)
    model = getattr(importlib.import_module(module_name), class_name)
    model = model().to(device)

    num_trainable_params = sum(p.numel() for p in model.yolo.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_trainable_params:,}")

    def dice_criterion(outputs, labels):
        labels_hot = to_one_hot(labels, num_classes).to(device)
        dice_score = dice(labels_hot, torch.nn.functional.softmax(outputs, dim=1))
        present_labels = torch.unique(labels)
        mask = torch.zeros(num_classes, device=device)
        mask[present_labels] = 1
        dice_masked = (mask * dice_score).sum() / len(present_labels)
        return 1 - dice_masked

    criterion = dice_criterion

    # Set YOLO to training mode
    model.yolo.train(data="imgs.yaml", epochs=10, imgsz=256)


    print("Starting training loop...")
    with torch.no_grad():
        dice_cum = 0
        for x_test, y_test in data_test_loader:
            output = model.predict(x_test.to(device))

            dice_score = dice_criterion(output, y_test.to(device))
            dice_cum += (1 - dice_score.item()) * x_test.size(0)
        print("valid dice ", dice_cum / len(data_test))
    print("Training complete.")

if __name__ == "__main__":
    main(model_name="sam_transformer")