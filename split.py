import os
import shutil

import torch
from model import resnet50
from test import create_segment
from utils import get_all_images, copy
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Path to the main folder
main_folder = "./data"
dst_dir = Path("D:/workspace/thesis/sources/mvfa-ad/data/Bone_AD")

# Abnormal images
heatmap_imgs = get_all_images("data/test-all")

# Normal images
normal_imgs = []
imgs = get_all_images("data/FracAtlas/XR_LEG")
normal_imgs.extend(imgs)
imgs = get_all_images("data/FracAtlas/XR_HAND")
normal_imgs.extend(imgs)

# Get indexes
idxes = [
    int(os.path.basename(img).replace("image", "").replace(".jpg", "")) for img in heatmap_imgs
]
# imgs2 = [img.split("/")[-1] for img in imgs2]
df = pd.read_csv("./data/bone-image-mapping.csv")

abnormal_imgs = []
for id in idxes:
    filtered_df = df[df["Index"] == id]
    img = filtered_df["Path"].tolist()[0]

    if not img:
        continue

    abnormal_imgs.append(img)

# Split
train_ungood_data, valid_ungood_data = train_test_split(abnormal_imgs, test_size=0.2, random_state=42)
train_good_data, valid_good_data = train_test_split(
    normal_imgs, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------
# load resnet model
net = resnet50(pretrained=True)
net.load_state_dict(torch.load("./ckpts/v0/Bone/best_model 85.pth.tar")["net"])
net = torch.nn.DataParallel(net)
net = net.cuda()
net.eval()

for image in tqdm(train_ungood_data, desc="Creating test abnormal data"):
    if "_negative" in image:
        name = os.path.basename(image)
        img_path = dst_dir / "test" / "Ungood" / "img" / name
        seg_path = dst_dir / "test" / "Ungood" / "anomaly_mask" / name

        os.makedirs(Path(img_path).parent, exist_ok=True)
        os.makedirs(Path(seg_path).parent, exist_ok=True)

        copy(image, img_path)
        create_segment(net, img_path, seg_path)

for image in tqdm(valid_ungood_data, desc="Creating valid abnormal data"):
    if "_negative" in image:
        name = os.path.basename(image)
        img_path = dst_dir / "valid" / "Ungood" / "img" / name
        seg_path = dst_dir / "valid" / "Ungood" / "anomaly_mask" / name

        os.makedirs(Path(img_path).parent, exist_ok=True)
        os.makedirs(Path(seg_path).parent, exist_ok=True)

        copy(image, img_path)
        create_segment(net, img_path, seg_path)

# ----------------------------------------------------------------

for image in tqdm(train_good_data, desc="Creating test normal data"):
    if "_positive" in image:
        name = os.path.basename(image)
        img_path = dst_dir / "test" / "good" / "img" / name
        seg_path = dst_dir / "test" / "good" / "anomaly_mask" / name

        # Open the image to get its size
        with Image.open(image) as img:
            width, height = img.size

        os.makedirs(Path(img_path).parent, exist_ok=True)
        os.makedirs(Path(seg_path).parent, exist_ok=True)

        copy(image, img_path)

        # Create a black (binary) image with the same size
        black_image = Image.new("1", (width, height), 0)  # '1' for binary mode, 0 for black
        black_image.save(seg_path)

for image in tqdm(valid_good_data, desc="Creating valid normal data"):
    if "_positive" in image:
        name = os.path.basename(image)
        img_path = dst_dir / "valid" / "good" / "img" / name
        seg_path = dst_dir / "valid" / "good" / "anomaly_mask" / name

        # Open the image to get its size
        with Image.open(image) as img:
            width, height = img.size

        os.makedirs(Path(img_path).parent, exist_ok=True)
        os.makedirs(Path(seg_path).parent, exist_ok=True)

        copy(image, img_path)

        # Create a black (binary) image with the same size
        black_image = Image.new(
            "1", (width, height), 0
        )  # '1' for binary mode, 0 for black
        black_image.save(seg_path)
