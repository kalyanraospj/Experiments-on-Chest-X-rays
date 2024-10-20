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


import numpy as np


def bbox_to_mask(image_size, bbox):
    """
    Convert a bounding box to a mask.

    Parameters:
        image_size (tuple): Size of the image (height, width).
        bbox (list): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
        numpy array: Binary mask with the bounding box region filled.
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)  # Create an empty mask

    x_min, y_min, x_max, y_max = bbox
    # Set the region inside the bounding box to 1 (or True)
    mask[y_min:y_max, x_min:x_max] = 1

    return mask


def mask_to_bbox(mask):
    """
    Convert a mask to a bounding box.

    Parameters:
        mask (numpy array): Binary mask.

    Returns:
        list: Bounding box [x_min, y_min, x_max, y_max].
    """
    # Find indices where mask is non-zero (i.e., where mask == 1)
    y_indices, x_indices = np.where(mask == 1)

    # If no object is found, return None or an empty box
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    # Get the bounding box coordinates from the min/max indices
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return [x_min, y_min, x_max, y_max]


def get_resize_bbox(annotations, file_path):
    image = cv2.imread(f"/content/chest_xray1/{file_path}")
    image_size = image.shape[:2]
    image = apply_clahe(image)
    image = cv2.resize(image, (1024, 1024))
    bboxes = annotations[file_path]
    boxes = []
    labels = []
    class_id_map = {"A": 1, "B": 2, "C": 3}

    if len(bboxes) > 0:
        for bbox in bboxes:
            class_id, center_x, center_y, width, height = bbox.split()
            center_x, center_y, width, height = map(
                float, [center_x, center_y, width, height]
            )
            # Convert to absolute coordinates
            x_min = (center_x - (width / 2)) * image_size[1]
            y_min = (center_y - (height / 2)) * image_size[0]
            x_max = (center_x + (width / 2)) * image_size[1]
            y_max = (center_y + (height / 2)) * image_size[0]

            bbox = map(int, [x_min, y_min, x_max, y_max])
            mask = bbox_to_mask(image_size, bbox)

            mask_1 = cv2.resize(mask, (1024, 1024))
            bbox_from_mask = mask_to_bbox(mask_1)
            if bbox_from_mask is None:
                print(file_path)
                print([x_min, y_min, x_max, y_max])
                continue

            boxes.append(bbox_from_mask)
            labels.append(class_id_map[class_id])
    else:
        boxes.append([0, 0, 1, 1])
        labels.append(0)
    return image, boxes, labels


def get_preprocessed_data(
    file_path="/content/drive/MyDrive/test_5c/chest_xray_images.json",
):
    data = json.load(open(file_path, "r"))
    annotations = {}
    for annot in data:
        annotations.update(annot)
    final_data = {}
    for i in annotations.keys():
        try:
            image, boxes, labels = get_resize_bbox(i)
            final_data.update({i: {"image": image, "boxes": boxes, "labels": labels}})
        except Exception as e:
            print(e)
            print(i)

    remove = []
    for i in final_data.keys():
        if len(final_data[i]["boxes"]) == 0:
            remove.append(i)

    for i in remove:
        final_data.pop(i)

    return final_data
