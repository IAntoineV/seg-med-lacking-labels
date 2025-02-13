import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from dataset import get_data
from torch.utils.data import DataLoader
from utils import to_one_hot
from metrics import dice

# Import segmentation_models_pytorch
import segmentation_models_pytorch as smp

num_classes = 55
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load your data (assumes get_data returns training and testing datasets)
data_train, data_test = get_data(test_size=0.05)
dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=2)
data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=2)

# Use SMP's U-Net with a ResNet34 encoder pretrained on ImageNet.
# This model accepts 1-channel input and outputs predictions for 55 classes.
model = smp.Unet(
    encoder_name="resnet34",  # Pretrained ResNet34 encoder
    encoder_weights="imagenet",  # Use ImageNet pretrained weights
    in_channels=1,  # Grayscale images of shape (1,256,256)
    classes=num_classes  # 55 segmentation classes
).to(device)

# Freeze the encoder to prevent overfitting with very few data.
for param in model.encoder.parameters():
    param.requires_grad = False

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {num_trainable_params:,}")


# Define a custom Dice-based criterion
def dice_criterion(outputs, labels):
    labels_hot = to_one_hot(labels, num_classes).to(device)
    dice_score = dice(labels_hot, torch.nn.functional.softmax(outputs, dim=1))
    present_labels = torch.unique(labels)
    mask = torch.zeros(num_classes, device=device)
    mask[present_labels] = 1
    mask[0] = 0  # Exclude background if needed
    dice_masked = (mask * dice_score).sum() / len(present_labels)
    return 1 - dice_masked


criterion = dice_criterion
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
num_epochs = 30

print("Starting training loop...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    it = 0
    for images, labels in dataloader:
        images = images.to(device)  # (B, 1, 256, 256)
        labels = labels.to(device)  # (B, 256, 256) with class indices
        optimizer.zero_grad()
        outputs = model(images)  # (B, 55, 256, 256)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        it += 1

    epoch_loss = running_loss / it
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation
    with torch.no_grad():
        dice_cum = 0
        for x_test, y_test in data_test_loader:
            output = model(x_test.to(device))
            # The dice_criterion returns (1 - dice), so we convert it back.
            dice_score = 1 - dice_criterion(output, y_test.to(device))
            dice_cum += dice_score.item() * x_test.size(0)
        valid_dice = dice_cum / len(data_test)
        print("Validation Dice Score: ", valid_dice)

print("Training complete.")
