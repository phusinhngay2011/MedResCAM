import tqdm
from utils import get_all_images, copy

from PIL import Image
import imagehash
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def find_best_match(ori_imgs, dst_img):
    dst_hash = imagehash.average_hash(Image.open(dst_img))
    best_match = None
    lowest_diff = float("inf")

    for ori_img in ori_imgs:
        ori_hash = imagehash.average_hash(Image.open(ori_img))
        diff = dst_hash - ori_hash

        if diff < lowest_diff:
            lowest_diff = diff
            best_match = ori_img

    return best_match


def match_images(ori_imgs, dst_imgs):
    matches = {}

    for dst_img in tqdm(dst_imgs):
        best_match = find_best_match(ori_imgs, dst_img)
        matches[dst_img] = best_match

    return matches

DST = "D:/workspace/thesis/sources/MedResCAM/data/test-all"
ORI = "D:/workspace/thesis/data/lqn"

ori_imgs = get_all_images(ORI)
dst_imgs = get_all_images(DST)

matches = match_images(ori_imgs, dst_imgs)
original = []
destination = []
for dst_img, ori_img in matches.items():
    original.append(ori_img)
    destination.append(dst_img)
    print(f"Destination Image: {dst_img} -> Original Image: {ori_img}")
    copy(dst_img, Path("data") / "meomeo" / os.path.basename(dst_img))
    copy(ori_img, Path("data") / "meomeo" / str(os.path.basename(dst_img).replace(".jpg", "_o.jpg")))

df = pd.DataFrame({"Original": original, "Dst": destination})

df.to_csv("bone_mapping_origin.csv")
