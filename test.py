import argparse
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from torchvision import transforms


# Define a transformation to resize the image to 240x240 and convert it to a tensor
transform = transforms.Compose(
    [
        # transforms.Resize((256, 256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def load_image(filepath):
    """Load an image, resize to 240x240, normalize pixel values to [0, 1], and binarize."""
    image = Image.open(filepath).convert("L")  # Convert to grayscale image (mode 'L')
    image = transform(image)  # Resize and convert to tensor
    image = (image > 0.5).float()  # Binarize the image
    return image


# def load_image(filepath):
#     """Load an image as a binary tensor, thresholding at 0.5."""
#     image = Image.open(filepath).convert("L")  # Convert to grayscale image (mode 'L')
#     image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
#     image = (image > 0.5).astype(np.float32)  # Binarize the image
#     return torch.from_numpy(image)


def calculate_iou(preds, labels):
    """Calculate the Intersection over Union (IoU) for binary images."""
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    iou = intersection / union
    return iou.item()


def calculate_mean_iou(gt_folder, pd_folder):
    """Calculate the mean IoU for all image pairs in the given folders."""
    gt_files = sorted(os.listdir(gt_folder))
    pd_files = sorted(os.listdir(pd_folder))

    if len(gt_files) != len(pd_files):
        raise ValueError(
            "The number of ground truth and predicted images must be the same."
        )

    ious = []
    for gt_file, pd_file in zip(gt_files, pd_files):
        if gt_file != pd_file:
            print(1)
        gt_path = os.path.join(gt_folder, gt_file)
        pd_path = os.path.join(pd_folder, pd_file)

        gt_image = load_image(gt_path)
        pd_image = load_image(pd_path)

        iou = calculate_iou(pd_image, gt_image)
        ious.append(iou)

    mean_iou = sum(ious) / len(ious)
    return mean_iou


def calculate_auc(gt_folder, pd_folder):
    """Calculate the AUC for all image pairs in the given folders."""
    gt_files = sorted(os.listdir(gt_folder))
    pd_files = sorted(os.listdir(pd_folder))

    if len(gt_files) != len(pd_files):
        raise ValueError(
            "The number of ground truth and predicted images must be the same."
        )

    gt_labels = []
    pd_probs = []

    for gt_file, pd_file in zip(gt_files, pd_files):
        gt_path = os.path.join(gt_folder, gt_file)
        pd_path = os.path.join(pd_folder, pd_file)

        gt_image = load_image(gt_path).flatten().numpy()
        pd_image = load_image(pd_path).flatten().numpy()

        gt_labels.extend(gt_image)
        pd_probs.extend(pd_image)

    auc = roc_auc_score(gt_labels, pd_probs)
    return auc


# Example usage
gt_folder = "data/{}_AD/valid/Ungood/anomaly_mask"
pd_folder = "outputs/{}_AD/v0/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default="Brain", type=str, help="Obj")
    args = parser.parse_args()

    print(args)
    gt_folder = gt_folder.format(args.obj)
    pd_folder = pd_folder.format(args.obj)
    mean_iou = calculate_mean_iou(gt_folder, pd_folder)
    print(f"Mean IoU: {mean_iou:.4f}")

    auc = calculate_auc(gt_folder, pd_folder)
    print(f"AUC: {auc:.4f}")
