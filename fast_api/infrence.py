import torch
from PIL import Image
import torchvision.transforms as T
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import json
import numpy as np


def apply_clahe(image):
    image = np.array(image)
    """Apply CLAHE to enhance X-ray image contrast."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image


def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)

    # Load the state dictionary into the model
    model.load_state_dict(
        torch.load(
            "/Users/kalyan/hack/5c/model_wights/faster_rcnn_model_15.pth",
            map_location=torch.device("cpu"),
        )
    )

    model.eval()  # Set to evaluation mode
    return model


def run_inference(model, image):
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Extract predictions
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    return boxes, labels, scores
