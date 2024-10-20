import torch
from PIL import Image
import torchvision.transforms as T
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import json
import numpy as np
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F
import argparse
from preprocessing import get_preprocessed_data


class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transform=None):
        """
        annotations: dict mapping image filenames to YOLO-style bounding box annotations
        img_dir: directory where the images are stored
        transform: torchvision transforms to apply to the images
        """
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_file = list(self.annotations.keys())[idx]
        image, boxes, labels = list(self.annotations[image_file].values())
        image = Image.fromarray(image)
        if len(boxes) < 0:
            boxes = [[0, 0, 1, 1]]
            labels = [3]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target


def evaluate(model, eval_loader, device):
    model.train()  # Temporarily set the model to training mode to compute losses
    eval_loss = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, targets in eval_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass to get the losses
            loss_dict = model(images, targets)

            # Ensure loss_dict is a dictionary
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                eval_loss += losses.item()

    avg_eval_loss = eval_loss / len(eval_loader)
    return avg_eval_loss


def collate_fn(batch):
    images = list(item[0] for item in batch)
    targets = [{k: v for k, v in t.items()} for t in list(item[1] for item in batch)]
    return images, targets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--showroom_config_folder", type=str, default="config/solar-tvs-nayandahalli"
    )
    parser.add_argument("--camera_idx", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set up TensorBoard writer
    writer = SummaryWriter("runs/faster_rcnn_xray_experiment_v2")

    data = get_preprocessed_data()

    keys = list(data.keys())
    random.shuffle(keys)

    augmentations = T.Compose(
        [
            T.RandomHorizontalFlip(
                p=0.5
            ),  # Flip the image horizontally with a 50% chance
            T.RandomRotation(degrees=10),  # Rotate the image by ±10 degrees
            T.ColorJitter(
                brightness=0.2, contrast=0.2
            ),  # Randomly change brightness/contrast
            T.ToTensor(),
            T.Normalize(
                mean=[0.5], std=[0.5]
            ),  # Normalize to zero mean and unit variance
        ]
    )
    train_ratio = 0.8
    train_split_index = int(len(keys) * train_ratio)

    train_keys = keys[:train_split_index]
    val_keys = keys[train_split_index:]

    train_data = {key: data[key] for key in train_keys}
    val_data = {key: data[key] for key in val_keys}

    # Now you have train_data and val_data dictionaries
    # You can create your datasets like this:
    train_dataset = XRayDataset(train_data, transform=augmentations)
    val_dataset = XRayDataset(
        val_data,
        transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]),
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, collate_fn=collate_fn
    )
    eval_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=True, collate_fn=collate_fn
    )

    device = "cuda"
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.train()
    model = model.to(device)

    # Set the optimizer, criterion, etc.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epochs
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        model.train()

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # Log losses to TensorBoard
            writer.add_scalar(
                "Loss/total_loss", losses.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Loss/classification_loss",
                loss_dict["loss_classifier"].item(),
                epoch * len(dataloader) + batch_idx,
            )
            writer.add_scalar(
                "Loss/box_regression_loss",
                loss_dict["loss_box_reg"].item(),
                epoch * len(dataloader) + batch_idx,
            )

        # Log average loss per epoch
        writer.add_scalar("Loss/avg_epoch_loss", epoch_loss / len(dataloader), epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)}")

        # Optional: Add learning rate logging
        for param_group in optimizer.param_groups:
            writer.add_scalar("Learning_Rate", param_group["lr"], epoch)

        # Evaluate model after each epoch
        avg_eval_loss = evaluate(model, eval_loader, device)
        writer.add_scalar("Eval_Loss/avg_epoch_loss", avg_eval_loss, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Eval Loss: {avg_eval_loss}")

        model_save_path = (
            f"/content/drive/MyDrive/test_5c/model/faster_rcnn_model_{epoch}.pth"
        )
        # Save the model's state dictionary
        torch.save(model.state_dict(), model_save_path)

    # Close the TensorBoard writer
    writer.close()
