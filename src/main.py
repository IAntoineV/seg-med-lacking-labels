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
    SamImageProcessor,
    SamVisionConfig,
    SamMaskDecoderConfig,
    SamPromptEncoderConfig,
)


def main(model_name="deeplabv3_resnet50", batch_size=16):
    # Load datasets
    data_train, data_test = get_data(test_size=0.05)

    dataloader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    data_test_loader = DataLoader(
        data_test, batch_size=batch_size, num_workers=2, drop_last=True
    )

    # Load DeepLab model
    cfg = model_configs[model_name]
    module_name, class_name = cfg.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    dlv3_model = model_class(pretrained=False, num_classes=num_classes).to(device)

    # Load SAM model and processor
    sam_checkpoint = "./cp/sam_vit_b_01ec64.pth"  # Path to SAM checkpoint
    vision_config = SamVisionConfig(
        # num_hidden_layers=1,
        # num_attention_heads=4,
        hidden_size=768,
    )
    mask_decoder_config = SamMaskDecoderConfig(
        # num_hidden_layers=1,
        # num_attention_heads=4,
    )
    prompt_encoder_config = SamPromptEncoderConfig()
    config = SamConfig(
        vision_config=vision_config,
        mask_decoder_config=mask_decoder_config,
        prompt_encoder_config=prompt_encoder_config,
    )
    checkpoint = torch.load(sam_checkpoint)
    sam_model = SamModel(config).to(device)

    # Adjust the keys to match Hugging Face's SamModel
    new_checkpoint = {}
    for key, value in checkpoint.items():
        if key.startswith("image_encoder"):
            new_key = key.replace("image_encoder", "vision_encoder")
        else:
            new_key = key
        new_checkpoint[new_key] = value

    # Load the adjusted checkpoint into the model
    sam_model.load_state_dict(checkpoint, strict=False)  # Load SAM checkpoint
    sam_model.eval()  # Ensure SAM model is in evaluation mode
    processor = SamImageProcessor(do_rescale=False, do_resize=False)

    # Set training parameters
    num_trainable_params = sum(
        p.numel() for p in dlv3_model.parameters() if p.requires_grad
    )
    print(f"Total trainable parameters for DeepLab: {num_trainable_params:,}")

    def dice_criterion(outputs, labels):
        labels_hot = to_one_hot(labels, num_classes).to(device)
        dice_score = dice(labels_hot, torch.nn.functional.softmax(outputs, dim=1))
        present_labels = torch.unique(labels)
        mask = torch.zeros(num_classes, device=device)
        mask[present_labels] = 1
        dice_masked = (mask * dice_score).sum() / len(present_labels)
        return 1 - dice_masked

    criterion = dice_criterion
    optimizer = optim.Adam(dlv3_model.parameters(), lr=1e-3)
    num_epochs = 30

    print("Starting training loop...")
    for epoch in tqdm(range(num_epochs)):
        dlv3_model.train()
        running_loss = 0.0
        it = 0
        for j, (images, labels) in tqdm(enumerate(dataloader)):
            if j > 4:
                break
            images = images.to(device)  # (B, 3, H, W)
            labels = labels.to(device)  # (B, H, W) with class indices
            optimizer.zero_grad()
            dlv3_outputs = dlv3_model(images)["out"]

            # Convert DeepLabV3 output to binary masks
            deep_masks = torch.argmax(dlv3_outputs, dim=1)  # (B, H, W)
            deep_masks = deep_masks.unsqueeze(1)  # (B, 1, H, W)

            with torch.no_grad():
                list_of_masks = []
                for i in range(deep_masks.size(0)):
                    inputs = processor(
                        images[i].permute(1, 2, 0).cpu().numpy(),
                        segmentation_maps=deep_masks[i].float().cpu().numpy(),
                        return_tensors="pt",
                    ).to(device)
                    sam_outputs = sam_model(**inputs)["pred_masks"][
                         :, :, 1, :, :
                    ]  # (1, num_masks, H, W)
                    refined_masks = torch.argmax(sam_outputs, dim=1)  # (1, 1, H, W)
                    list_of_masks.append(refined_masks[0])
                refined_masks = torch.stack(list_of_masks, dim=0)
            refined_masks_one_hot = (
                to_one_hot(refined_masks, num_classes).to(device).type(torch.float32)
            )  # (B, num_classes, H, W)
            loss_dlv3 = criterion(dlv3_outputs, labels)  # Loss for DeepLabV3 outputs
            loss_sam = criterion(
                refined_masks_one_hot, labels
            )  # Loss for SAM refined masks
            loss = alpha * loss_dlv3 + (1 - alpha) * loss_sam
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
                load_batch_size = x_test.size(0)
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                dlv3_outputs = dlv3_model(x_test)["out"]
                deep_masks = torch.argmax(dlv3_outputs, dim=1)  # (B, H, W)
                deep_masks = deep_masks.unsqueeze(1)  # (B, 1, H, W)

                list_of_masks = []
                for i in range(deep_masks.size(0)):
                    inputs = processor(
                        images[i].permute(1, 2, 0).cpu().numpy(),
                        segmentation_maps=deep_masks[i].float().cpu().numpy(),
                        return_tensors="pt",
                    ).to(device)
                    sam_outputs = sam_model(**inputs)["pred_masks"][
                        :, :, 1, :, :
                    ]  # (B, num_masks, H, W)
                    refined_masks = torch.argmax(sam_outputs, dim=1)  # (B, H, W)
                    list_of_masks.append(refined_masks[0])
                refined_masks = torch.stack(list_of_masks, dim=0)
                refined_masks_one_hot = (
                    to_one_hot(refined_masks, num_classes)
                    .to(device)
                    .type(torch.float32)
                )  # (B, num_classes, H, W)

                # Calculate Dice score, ignoring padded elements
                dice_score = dice_criterion(
                    refined_masks_one_hot[:batch_size], y_test[:batch_size]
                )
                dice_cum += (1 - dice_score.item()) * batch_size

            print(f"Validation Dice: {dice_cum / len(data_test):.4f}")

    print("Training complete.")


if __name__ == "__main__":
    alpha = 0.3
    num_classes = 55
    batch_size = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_configs = {
        "deeplabv3_mobilenet_v3_large": "torchvision.models.segmentation.deeplabv3_mobilenet_v3_large",
        "deeplabv3_resnet50": "torchvision.models.segmentation.deeplabv3_resnet50",
    }
    main(model_name="deeplabv3_mobilenet_v3_large", batch_size=batch_size)
