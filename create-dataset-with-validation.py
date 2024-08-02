import copy
import csv
import json
import os
from email.mime import image
from pathlib import Path
from typing import Tuple
from xml.dom.expatbuilder import theDOMImplementation

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import resnet50
from utils import get_all_images, copy

Trans = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

Normal = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)

bce = torch.nn.BCEWithLogitsLoss()


def generate_grad_cam(net, ori_image):
    """
    :param net: deep learning network(ResNet DataParallel object)
    :param ori_image: the original image
    :return: gradient class activation map
    """
    input_image = Trans(ori_image)

    feature = None
    gradient = None

    def forward_hook(module, input, output):
        nonlocal feature
        feature = output.data.cpu().numpy()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].data.cpu().numpy()

    # print(net.module)
    net.module.layer4.register_forward_hook(forward_hook)
    net.module.layer4.register_full_backward_hook(backward_hook)

    out = net(input_image.unsqueeze(0))

    pred = out.data > 0.5

    net.zero_grad()

    loss = bce(out, pred.float())
    loss.backward()

    feature = np.squeeze(feature, axis=0)
    gradient = np.squeeze(gradient, axis=0)

    weights = np.mean(gradient, axis=(1, 2), keepdims=True)

    cam = np.sum(weights * feature, axis=0)

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = 1.0 - cam
    cam = np.uint8(cam * 255)
    return cam


def localize(cam_feature, ori_image):
    """
    localize the abnormality region using grad_cam feature
    :param cam_feature: cam_feature by generate_grad_cam
    :param ori_image: the original image
    :return: img with heatmap, the abnormality region is highlighted
    """
    ori_image = np.array(ori_image)
    activation_heatmap = cv2.applyColorMap(cam_feature, cv2.COLORMAP_JET)
    activation_heatmap = cv2.resize(
        activation_heatmap, (ori_image.shape[1], ori_image.shape[0])
    )
    img_with_heatmap = 0.15 * np.float32(activation_heatmap) + 0.85 * np.float32(
        ori_image
    )
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap) * 255
    return img_with_heatmap


def heatmap2segment(cam_feature, ori_image, threshold=0.8):
    ori_image = np.array(ori_image)
    cam_feature = cv2.resize(cam_feature, (ori_image.shape[1], ori_image.shape[0]))

    crop = np.uint8(cam_feature > threshold * 255)

    (totalLabels, label_ids, values, centroid) = (
        cv2.connectedComponentsWithStatsWithAlgorithm(crop, 4, cv2.CV_32S, ccltype=1)
    )
    # print(
    #     f"totalLabels: {totalLabels}, label_ids: {label_ids}, values: {values}, centroid: {centroid}"
    # )

    output = np.zeros(ori_image.shape, dtype="uint8")

    # Loop through each component
    for i in range(1, totalLabels):
        componentMask = (label_ids == i).astype("uint8") * 255
        output = cv2.bitwise_or(output, componentMask)
    output = Image.fromarray(output).convert("RGB")

    return output


def create_abnormal_boundary_line(net, img_path, save_path, threshold=[0.8]):
    # Load and preprocess the original image
    ori_image = Image.open(img_path).convert("RGB")
    # Assuming these are functions you have defined elsewhere
    cam_feature = generate_grad_cam(net, ori_image)
    heatmap = localize(cam_feature.copy(), ori_image.copy())

    # Create a list of images to concatenate horizontally
    imgs = [ori_image, Image.fromarray(np.uint8(heatmap)).convert("RGB")]

    for thes in threshold:
        heatmap_with_contours = heatmap.copy()
        segment = heatmap2segment(cam_feature, ori_image.convert("L"), thes)
        segment = segment.convert("L")
        segment = np.array(segment)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(heatmap_with_contours, contours, -1, (0, 0, 255), 10)

        # Assuming 'segment' is defined elsewhere in your code
        segment = Image.fromarray(np.uint8(heatmap_with_contours)).convert(
            "RGB"
        )  # Convert heatmap to RGB for PIL

        # Create a list of images to concatenate horizontally
        imgs.append(segment)

    # Calculate dimensions for the combined image
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image to store the combined result
    result = Image.new("RGB", (total_width, max_height))

    # Paste each image into the combined result side by side
    x_offset = 0
    for im in imgs:
        result.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # Ensure the directory exists where the image will be saved
    os.makedirs(Path(save_path).parent, exist_ok=True)

    # Save the combined image

    result = np.array(result)
    # Calculate the dimensions for the borders
    height, width = result.shape[:2]
    top_border = int(height / 7.8)  # Top 1/5 of the image
    bottom_border = height - top_border  # Bottom 1/5 of the image

    # Add black borders to the top and bottom of the image
    result_with_borders = np.zeros_like(result)
    result_with_borders[top_border:bottom_border, :, :] = result[
        top_border:bottom_border, :, :
    ]

    cv2.imwrite(
        str(save_path), np.array(result_with_borders), [cv2.IMWRITE_JPEG_QUALITY, 100]
    )


def create_segment(net, img_path, seg_path, threshold=0.8):
    ori_image = Image.open(img_path).convert("RGB")
    cam_feature = generate_grad_cam(net, ori_image)
    segment = heatmap2segment(cam_feature, ori_image.convert("L"), threshold)
    os.makedirs(Path(seg_path).parent, exist_ok=True)
    segment.save(seg_path)


def predict(net, ori_image):
    input_image = Trans(ori_image)
    # Add a Batch Dimension and Move to GPU:
    input_image = input_image.unsqueeze(0).cuda()
    outputs = net(input_image)
    preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
    return preds


# false -> abnormal
# true -> normal


def predict_and_create_segment(net, img_path, threshold=0.8):
    ori_image = Image.open(img_path).convert("RGB")
    pred = predict(net, ori_image)

    # Abnormal
    if torch.equal(pred, torch.zeros_like(pred)):
        cam_feature = generate_grad_cam(net, ori_image)
        heatmap = localize(cam_feature.copy(), ori_image.copy())
        segment = heatmap2segment(cam_feature, ori_image.convert("L"), threshold)
        return (
            False,
            ori_image,
            Image.fromarray(np.uint8(heatmap)).convert("RGB"),
            segment,
        )
    # Normal
    else:
        return (True, None, None, None)


def create_path_dict(csv_path):
    path_dict = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            index = int(row["Index"])
            formatted_image_name = "image{:04}.jpg".format(index)
            path_dict[formatted_image_name] = row["Path"]
    return path_dict


if __name__ == "__main__":
    visualize_path = Path("./data/test-images")
    ds_path = Path("D:/workspace/thesis/sources/mvfa-ad/data/Bone_v3_AD")
    threshold = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Load the ResNet50 model
    net = resnet50(pretrained=True)
    net.load_state_dict(torch.load("./ckpts/v0/Bone/best_model 85.pth.tar")["net"])
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.eval()

    # Load images for testing
    imgs = get_all_images("./data/test-all")

    with open("evaluations/BV Nguyen Tri Phuong.json", "r") as f:
        dt1 = json.load(f)

    with open("evaluations/BS NHAN.json", "r") as f:
        dt2 = json.load(f)

    imgs = [
        img
        for img in imgs
        if dt1[os.path.basename(img)]["good"] == "True"
        and dt2[os.path.basename(img)]["good"] == "True"
    ]

    bone_map = create_path_dict("./data/bone-image-mapping.csv")
    imgs = [bone_map[os.path.basename(img)] for img in imgs]

    # Get all abnormal images that predict correctly
    result = []
    for i, img_path in tqdm(enumerate(imgs), desc="Predicting "):
        # Create boundary line around abnormal regions
        filename = os.path.basename(img_path)
        save_path = visualize_path / filename

        ori_image = Image.open(img_path).convert("RGB")
        pred = predict(net, ori_image)
        # Abnormal
        if torch.equal(pred, torch.zeros_like(pred)):
            # create_abnormal_boundary_line(
            #     net, img_path, save_path=save_path, threshold=threshold
            # )
            result.append(img_path)
        else:
            print("Wrong")
            pass 

    valid = result[:16]
    test = result[16:]

    for img in tqdm(valid, desc="Add to train "): 
        filename = os.path.basename(img)
        valid_path = ds_path / "valid" / "Ungood"
        img_path = valid_path / "img" / filename
        seg_path = valid_path / "anomaly_mask" / filename
        create_segment(net, img, seg_path)
        copy(img, img_path)

    for img in tqdm(test, desc="Add to test "): 
        filename = os.path.basename(img)
        test_path = ds_path / "test" / "Ungood"
        img_path = test_path / "img" / filename
        seg_path = test_path / "anomaly_mask" / filename
        create_segment(net, img, seg_path)
        copy(img, img_path)
