from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    scale_accross_batch_and_channels,
    scale_cam_image,
)

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    scale_accross_batch_and_channels,
    scale_cam_image,
)
import requests
import torchvision.transforms as T
import torchvision
from PIL import Image


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]["labels"].cpu().numpy()]
    pred_labels = outputs[0]["labels"].cpu().numpy()
    pred_scores = outputs[0]["scores"].detach().cpu().numpy()
    pred_bboxes = outputs[0]["boxes"].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
        )
        cv2.putText(
            image,
            classes[i],
            (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return image


coco_names = ["negative", "Path_1", "Path_2", "Path_3"]


# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def fasterrcnn_reshape_transform(x):
    target_size = x["pool"].size()[-2:]
    activations = []
    for key, value in x.items():
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(value), target_size, mode="bilinear"
            )
        )
    activations = torch.cat(activations, axis=1)
    return activations


class FasterRCNNBoxScoreTarget:
    """For every original detected bounding box specified in "bounding boxes",
    assign a score on how the current bounding boxes match it,
      1. In IOU
      2. In the classification score.
    If there is not a large enough overlap, or the category changed,
    assign a score of 0.

    The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if (
                ious[0, index] > self.iou_threshold
                and model_outputs["labels"][index] == label
            ):
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output


def get_grad_cam(image, model, labels, boxes):

    image_float_np = np.float32(image) / 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.5], std=[0.5]
            ),  # Normalize to zero mean and unit variance
        ]
    )
    print(image.shape)
    input_tensor = transform(Image.fromarray(image))
    input_tensor = input_tensor.unsqueeze(0)

    boxes, classes, labels, indices = predict(input_tensor, model, "cpu", 0.1)

    target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(model, target_layers, reshape_transform=fasterrcnn_reshape_transform)

    grayscale_cam = cam(input_tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    # prompt: convert gray 2 rgb

    grayscale_cam_rgb = cv2.cvtColor(grayscale_cam, cv2.COLOR_GRAY2RGB)
    image_float_np_rgb = cv2.cvtColor(image_float_np, cv2.COLOR_GRAY2RGB)

    cam_image = show_cam_on_image(image_float_np_rgb, grayscale_cam_rgb, use_rgb=True)
    # And lets draw the boxes again:
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    return Image.fromarray(image_with_bounding_boxes)
