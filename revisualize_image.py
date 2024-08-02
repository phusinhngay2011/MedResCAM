import os
from pathlib import Path
from test import create_abnormal_boundary_line

import pandas as pd
import torch
from tqdm import tqdm

from model import resnet50
from utils import copy, get_all_images

dst_dir = Path(
    "D:/workspace/thesis/sources/phusinhngay2011.github.io/assets/images/Bones/visualize-bone"
)
selected_imgs = get_all_images("./data/test-all")

idxes = [
    int(os.path.basename(img).replace("image", "").replace(".jpg", ""))
    for img in selected_imgs
]

# imgs2 = [img.split("/")[-1] for img in imgs2]
df = pd.read_csv("./data/bone-image-mapping.csv")

abnormal_imgs = []
for id in idxes:
    filtered_df = df[df["Index"] == id]
    img = filtered_df["Path"].tolist()[0]

    if not img:
        continue

    abnormal_imgs.append({"path": img, "index": id})

#  load resnet model
net = resnet50(pretrained=True)
net.load_state_dict(
    torch.load(
        "D:/workspace/thesis/sources/MedResCAM/ckpts/v0/Bone/best_model 85.pth.tar"
    )["net"]
)
net = torch.nn.DataParallel(net)
net = net.cuda()
net.eval()

for image in tqdm(abnormal_imgs):
    if "_negative" in image["path"]:
        origin_path = image["path"]

        save_path = dst_dir / "image{:04}.jpg".format(image["index"])

        create_abnormal_boundary_line(net, origin_path, save_path)
