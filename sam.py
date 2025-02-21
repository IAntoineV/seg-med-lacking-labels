import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from ultralytics import YOLO

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "src\cp\sam_vit_b_01ec64.pth"

class YOLOv8SAM(torch.nn.Module):
    def __init__(self, model_name):
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
        self.mask_predictor = SamPredictor(sam)
        self.yolo = YOLO(model_name)
        self.yolo.fuse()

    def predict(self, image):
        detections = self.yolo.predict(image)
            # Check if there are fish detections
        if len(detections[0].boxes) == 0:
            return
        # Run frame and detections through SAM to get masks
        transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(
            detections[0].boxes.xyxy, [image.shape[-2], image.shape[-1]]
        )
        self.mask_predictor.set_image(image)
        masks, _, _ = self.mask_predictor.predict_torch(
            boxes=transformed_boxes,
            multimask_output=False,
            point_coords=None,
            point_labels=None
        )
        return masks