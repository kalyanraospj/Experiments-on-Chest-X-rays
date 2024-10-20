
# Faster R-CNN Training and Grad-CAM Visualization

This repository contains the code to train a Faster R-CNN model on X-ray images (with positive and negative samples) and to visualize the regions of interest using Grad-CAM (Gradient-weighted Class Activation Mapping). The repository also includes preprocessing techniques like CLAHE (Contrast Limited Adaptive Histogram Equalization) and augmentation to enhance model performance and robustness.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Requirements](#requirements)
- [Usage](#usage)

## Dataset

The dataset used for training consists of X-ray images with annotations in YOLO format. The dataset contains:
- **Positive Samples**: Images with bounding boxes representing pathologies.
- **Negative Samples**: Images without any bounding boxes.

Example of the annotation format:
- **Positive Sample**: \`"2023_04_10_20391E70_AA5E5ED2_09A6C2EE.jpeg": ["C 0.2158 0.4024 0.0296 0.0267"]\`
- **Negative Sample**: \`"2022_10_01_25F62808_B16FAE6B_1DBA457F.jpeg": []\`

## Preprocessing

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
   - CLAHE is applied to enhance the contrast of the X-ray images, making the pathologies more visible for both human and machine interpretation.

2. **Bounding Box Handling**:
   - The annotations (bounding boxes in YOLO format) are transformed when resizing the images.
   - Bounding boxes are converted to masks and vice versa during certain stages of training.

3. **Normalization & Augmentation**:
   - Various augmentations like flipping, rotation, and resizing are applied to improve model robustness and generalization.

## Model Training

We use the **Faster R-CNN** model (\`fasterrcnn_resnet50_fpn\`) for object detection and localization.

### Training Process
- **Data Loader**: The dataset is loaded with bounding box annotations, and a custom data loader is used to handle image resizing and augmentation.
- **Negative and Positive Samples**: The model is trained on both negative and positive samples. Negative samples contribute to learning what **non-pathological** images look like.
- **Evaluation**: An evaluation data loader is included to compute evaluation loss and measure the model's performance on unseen data.

### Loss Functions
The loss during training includes both **classification loss** and **localization loss**. During the evaluation stage, we calculate the average loss over the dataset.

### TensorBoard Integration
Training metrics (like loss, accuracy, etc.) are tracked using TensorBoard for real-time visualization.

## Model Evaluation

During the evaluation stage:
1. The model is set to evaluation mode.
2. The validation dataset is used to compute the average evaluation loss.
3. The loss consists of the same components as during training (classification and bounding box regression).

### Example Evaluation Code:

```python
def evaluate(model, eval_loader, device):
    model.eval()  # Set the model to evaluation mode
    eval_loss = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, targets in eval_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass to get the losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            eval_loss += losses.item()

    avg_eval_loss = eval_loss / len(eval_loader)
    return avg_eval_loss
```

## Grad-CAM Visualization

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is applied to visualize the regions of the X-ray images that the Faster R-CNN model focuses on when detecting pathologies.

### Steps for Grad-CAM:
1. Select a layer in the Faster R-CNN model to compute the gradients (typically the last convolutional layer).
2. Compute the gradients of the target class with respect to the selected feature maps.
3. Generate the Grad-CAM heatmap and overlay it on the original image.

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Matplotlib
- Torchvision
- TensorBoard

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Faster R-CNN Model:

1. Clone the repository:
 ```bash
   git clone https://github.com/your-repo/fasterrcnn-xray
   cd fasterrcnn-xray
```

2. Prepare the dataset and update the dataset paths in the config or script.

3. Run the training script:
   ```bash
   python train.py --epochs 50 --batchsize 8
   ```

4. Monitor training in TensorBoard:
   ```bash
   tensorboard --logdir=runs
   ```


## Running Streamlit application
```bash
streamlit run app.py
```


# Summary


## Approach to X-ray Pathology Detection

The objective of this experiment was to train a Faster R-CNN model for detecting pathologies in X-ray images, including both positive (images with pathologies) and negative (images without pathologies) samples. This detection was achieved by leveraging region proposal networks to identify candidate regions of interest in the X-ray images and applying classification and bounding box regression to detect and localize pathologies.

### Key Steps:

1. **Preprocessing**:
   - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) was applied to improve contrast and enhance the visibility of pathologies.
   - Bounding boxes provided in YOLO format were normalized, transformed, and resized accordingly.
   - Data augmentation (flipping, rotation, resizing) was used to improve model robustness and handle the variability in pathology appearance.

2. **Model Training**:
   - **Faster R-CNN with ResNet-50 backbone**: This architecture was chosen due to its success in object detection tasks. The model was trained to detect pathologies, using both negative and positive samples.
   - Losses were computed for both **classification** (presence of pathology) and **bounding box regression** (localization of pathology).

3. **Evaluation**:
   - After training, the model was evaluated on a validation dataset using metrics such as **mAP** (mean Average Precision) and **IoU** (Intersection over Union). These metrics provide a good understanding of the model’s ability to correctly classify and localize pathologies.

4. **Grad-CAM**:
   - **Grad-CAM (Gradient-weighted Class Activation Mapping)** was applied to visualize the regions in the X-ray images where the model focused when detecting pathologies. This helped in understanding model behavior and interpreting its predictions.


## Mogel Metrics

```
                  Class    Images  Instances       P          R         mAP50     mAP50-95
                   all        642       1363      0.227      0.247      0.217      0.108
              negative        276        276          0          0          0          0
                 path1        150        649      0.258      0.108     0.0961      0.025
                 path2        105        105      0.447      0.771      0.711      0.388
                 path3        111        333      0.202      0.108     0.0596     0.0178


```


## Challenges Encountered and Solutions

1. **Class Imbalance (Few Positive Samples)**:
   - X-ray datasets often have far more negative samples than positive ones, leading to class imbalance during training.
   - **Solution**: Over-sampling of positive samples and targeted data augmentation (rotation, flipping) were applied to balance the dataset.

2. **Small and Subtle Pathologies**:
   - Many pathologies in X-ray images are small and subtle, making them challenging for the model to detect accurately due to its architectural limitations.
   - **Solution**: Enhance the training data with precise annotations and apply additional augmentation to images containing small pathologies to increase the model's sensitivity to subtle abnormalities. 

3. **Bounding Box Regression for Small Anomalies**:
   - Localizing small pathologies with bounding boxes proved challenging, especially when the bounding box size was very small.
   - **Solution**: Careful tuning of anchor box scales and aspect ratios was done to improve the model’s ability to detect and localize small pathologies. Consider experimenting with different architectures, such as YOLO, which demonstrates high mAP performance for small bounding boxes.

4. **Generalization to Different Pathologies**:
   - The model's performance varied across different types of pathologies, with more distinct or large pathologies being easier to detect.
   - **Solution**: Additional training with diverse datasets and better data augmentation helped improve generalization across various pathology types.

## Potential Improvements and Future Work

1. **Integration of Additional Preprocessing Techniques**:
   - While CLAHE was effective in improving contrast, further preprocessing techniques (such as edge detection or noise reduction) could help improve model performance for subtle or small pathologies.
   
2. **Multi-Scale Training**:
   - To further improve the model’s ability to detect small pathologies, multi-scale training could be employed, where the model is trained on images of different scales to handle the varying sizes of pathologies better.

3. **Focal Loss for Handling Class Imbalance**:
   - Implementing **Focal Loss** could help in handling the imbalance between positive and negative samples, reducing the impact of easy negatives and focusing the model's attention on harder, more ambiguous cases.

4. **Ensemble Models**:
   - Combining predictions from multiple models (e.g., Faster R-CNN, YOLO) through ensemble learning could improve detection accuracy by leveraging the strengths of different architectures.

5. **Transfer Learning with Pretrained Models on Medical Imaging**:
   - Using pre-trained models that are already fine-tuned on medical imaging datasets could enhance the accuracy of detecting medical pathologies. Fine-tuning these models specifically for X-ray pathology detection could yield better performance.

6. **Active Learning for Hard Example Mining**:
   - Incorporating an active learning framework where the model focuses more on hard-to-detect pathologies in each training epoch could help improve the detection of challenging pathologies.




Note:

Due to the time constraints of this assessment and the need to meet my client's deliverables, I am unable to fully optimize the model or explore new architectures within the limited timeframe. However, I have implemented the training and inference pipeline using Streamlit and FastAPI. This project has been a refreshing and intense experience, much like a hackathon, and it was exciting to work under such tight deadlines again after a long time.
