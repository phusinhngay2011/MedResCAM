import csv
import json
import os
import test
from pathlib import Path
from test import create_segment

import imagehash
import pandas as pd
from matplotlib.pyplot import get
from PIL import Image, ImageDraw
from tqdm import tqdm

from utils import copy, get_all_images

bone_v1_path = Path("D:/workspace/thesis/sources/mvfa-ad/data/Bone_AD")
bone_v2_path = Path("D:/workspace/thesis/sources/mvfa-ad/data/Bone_v2_AD")

bone_v1 = get_all_images(bone_v1_path)

bone_v2 = [img.replace("Bone_AD", "Bone_v2_AD") for img in bone_v1]

for img in bone_v1:
    if "Ungood" not in img:
        copy(Path(img), Path(img.replace("Bone_AD", "Bone_v2_AD")))

bone_v1 = [path for path in bone_v1 if "anomaly_mask" not in path and "Ungood" in path]


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


def create_binary_image(original_img, vertices):
    width, height = original_img.size

    # Create a blank (black) image
    binary_img = Image.new("L", (width, height), 0)

    # Create a draw object
    draw = ImageDraw.Draw(binary_img)

    # Draw the polygon and fill it with white (value 255)
    draw.polygon(vertices, outline=255, fill=255)

    return binary_img


def create_path_dict(csv_path):
    path_dict = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            index = int(row["Index"])
            formatted_image_name = "image{:04}.jpg".format(index)
            path_dict[formatted_image_name] = row["Path"]
    return path_dict


# Example usage
bone_mapping_csv_path = "./data/bone-image-mapping.csv"
path_dict = create_path_dict(bone_mapping_csv_path)

false_imgs_path = "./test.csv"
false_imgs = pd.read_csv(false_imgs_path)["normal"].tolist()
# Load JSON data from file
with open("data/evaluations (12).json", "r") as f:
    json_data = json.load(f)

selected_imgs = json_data.keys()
idxes = [
    int(os.path.basename(img).replace("image", "").replace(".jpg", ""))
    for img in selected_imgs
]

changes = {
    "bone_v1": [],
    "modified": [],
}
name = []
good = []
bad = []
false_true = []
false_false = []
# Process each image and draw polygons
for image, data in tqdm(json_data.items()):
    name.append(image)
    if (
        "polygon" not in data
        or len(data["polygon"]["all_vertex_x"]) < 2
        or len(data["polygon"]["all_vertex_y"]) < 2
        or data["good"] == True
    ):
        good.append(1)
        bad.append(0)
        if image in false_imgs:
            print("False image found in good list: ", image)
            false_true.append(1)
            false_false.append(0)
        else:
            false_true.append(0)
            false_false.append(0)

        image_path = path_dict[image]
        best_match = find_best_match(bone_v1, image_path)
        if best_match is None:
            print("No match found for image ", image_path)
            continue
        copy(Path(best_match), Path(best_match.replace("Bone_AD", "Bone_v2_AD")))
        copy(
            Path(best_match.replace("/img/", "/anomaly_mask/")),
            Path(
                best_match.replace("/img/", "/anomaly_mask/").replace(
                    "Bone_AD", "Bone_v2_AD"
                )
            ),
        )
    else:
        good.append(0)
        bad.append(1)
        if image in false_imgs:
            print("False image found in bad list: ", image)
            false_true.append(0)
            false_false.append(1)
        else:
            false_true.append(0)
            false_false.append(0)
        continue
        image_path = path_dict[image]

        best_match = find_best_match(bone_v1, image_path)
        if best_match is None:
            print("No match found for image ", image_path)
            continue

        original_img = Image.open(best_match)
        width, height = original_img.size

        vertices_x = [x - 2 * width for x in data["polygon"]["all_vertex_x"]]
        vertices_y = data["polygon"]["all_vertex_y"]
        vertices = list(zip(vertices_x, vertices_y))

        segment = create_binary_image(original_img, vertices)
        segment.save(
            best_match.replace("Bone_AD", "Bone_v2_AD").replace(
                "/img/", "/anomaly_mask/"
            )
        )
        changes["bone_v1"].append(best_match)
        changes["modified"].append(image_path)
        # draw_polygon(abnormal_imgs[image_path], vertices_x, vertices_y)

# pd.DataFrame(changes).to_csv("changes.csv")
pd.DataFrame(
    {
        "name": name,
        "good": good,
        "bad": bad,
        "false_true": false_true,
        "false_false": false_false,
    }
).to_csv("./reanalysis.csv", index=False)
