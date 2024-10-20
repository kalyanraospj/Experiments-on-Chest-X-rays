# app.py
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from grad_cam.fast_rcnn import get_grad_cam
import cv2


def apply_clahe(image):
    image = np.array(image)
    """Apply CLAHE to enhance X-ray image contrast."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image


# Load the pre-trained Faster R-CNN model
@st.cache(allow_output_mutation=True)
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)

    # Load the state dictionary into the model
    model.load_state_dict(
        torch.load(
            "model_wights/faster_rcnn_model_15.pth", map_location=torch.device("cpu")
        )
    )

    model.eval()  # Set to evaluation mode
    return model


# Perform object detection
def run_inference(model, image):
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return predictions


# Draw bounding boxes on image
def draw_boxes(image, boxes, labels, scores, threshold=0.1):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i in range(len(boxes)):
        print(scores[i])
        if scores[i] >= threshold:

            box = boxes[i]
            label = labels[i]
            score = scores[i]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=6,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            plt.text(
                box[0],
                box[1],
                f"{label}: {score:.2f}",
                color="white",
                bbox=dict(facecolor="red", alpha=0.5),
            )
    st.pyplot(fig)


def draw_boxes_gradcam(image, model, labels, boxes):
    fig, ax = plt.subplots(1)
    image = get_grad_cam(image, model, labels, boxes)
    ax.imshow(image)
    st.pyplot(fig)


# Streamlit app
def main():
    st.title("Object Detection using Faster R-CNN")
    st.write("Upload an image and run detection inference.")

    # Load model
    model = load_model()

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cla = apply_clahe(image)
        image = Image.fromarray(image_cla)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Inference"):
            st.write("Running detection inference...")

            # Run inference
            predictions = run_inference(model, image)

            # Extract predictions
            boxes = predictions["boxes"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()

            # Map labels to COCO class names (if needed)
            COCO_INSTANCE_CATEGORY_NAMES = [
                "nagative",
                "Path_1",
                "Path_2",
                "Path_3",
            ]

            labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels]

            # Draw boxes
            st.write("detection Output.")
            draw_boxes(image=image, boxes=boxes, labels=labels, scores=scores)
            st.write("Grad Cam Output")
            draw_boxes_gradcam(image=image_cla, model=model, labels=labels, boxes=boxes)


if __name__ == "__main__":
    main()
