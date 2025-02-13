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

# Import SAM and related models from Hugging Face Transformers
from transformers import (
    SamModel,
    SamConfig,
    SamVisionConfig,
    SamMaskDecoderConfig,
    SamPromptEncoderConfig,
    SamImageProcessor,
    DetrImageProcessor,
    DetrForObjectDetection,

)

num_classes = 55
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_configs = {
    "deeplabv3_mobilenet_v3_large": "torchvision.models.segmentation.deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50": "torchvision.models.segmentation.deeplabv3_resnet50",
    "sam_transformer": "transformers.SamModel",
}

def main(model_name="deeplabv3_mobilenet_v3_large"):
    data_train, data_test = get_data(test_size=0.05)

    dataloader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=2)

    if model_name == "sam_transformer":

        vision_config = SamVisionConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
            hidden_size=128,
        )
        mask_decoder_config = SamMaskDecoderConfig(
            num_hidden_layers=1,
            num_attention_heads=4,
        )

        prompt_encoder_config = SamPromptEncoderConfig()
        config = SamConfig(
            vision_config=vision_config,
            mask_decoder_config=mask_decoder_config,
            prompt_encoder_config=prompt_encoder_config,
        )

        model = SamModel(config).to(device)
        processor = SamImageProcessor(
            do_rescale=False,
            do_resize=False,
        )
        
        cfg = model_configs["deeplabv3_mobilenet_v3_large"]
        module_name, class_name = cfg.rsplit(".", 1)
        dlv3_model_class = getattr(importlib.import_module(module_name), class_name)
        dlv3_model = dlv3_model_class(pretrained=False, num_classes=num_classes).to(device)
    else:
        cfg = model_configs[model_name]
        module_name, class_name = cfg.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = model_class(pretrained=False, num_classes=num_classes).to(device)
        processor = None

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
            if model_name == "sam_transformer":
                boxes = dlv3_model(images)["out"]
                print(boxes)
                inputs = processor(images, segmentation_maps=boxes, return_tensors="pt").to(device)
                outputs = model(**inputs)["pred_masks"]

            else:
                model(images)["out"]
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
                if model_name == "sam_transformer":
                    inputs = processor(x_test.to(device), return_tensors="pt").to(
                        device
                    )
                    output = model(**inputs)["pred_masks"]
                    
                else:
                    output = model(x_test.to(device))["out"]

                dice_score = dice_criterion(output, y_test.to(device))
                dice_cum += (1 - dice_score.item()) * x_test.size(0)
            print("valid dice ", dice_cum / len(data_test))
    print("Training complete.")

if __name__ == "__main__":
    main(model_name="deeplabv3_resnet50")