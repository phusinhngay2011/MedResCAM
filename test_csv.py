import pandas as pd
import os
from utils import copy, get_all_images
from pathlib import Path
from tqdm import tqdm

# df = pd.read_csv("data/mura/train_image_paths.csv", header=None)

# for i, row in df.iterrows():
#     row[0] = row[0].replace("MURA-v1.1", "mura")
#     if "negative" in row[0]:
#         copy(
#             Path(os.path.join("data", row[0])),
#             Path(os.path.join("data", "mura-inspect", "negative", f"{i}.png")),
#         )
#     else:
#         copy(
#             Path(os.path.join("data", row[0])),
#             Path(os.path.join("data", "mura-inspect", "positive", f"{i}.png")),
#         )

images = get_all_images("./data/mura")

for i, img in tqdm(enumerate(images)):
    if "negative" in img:
        copy(
            Path(img),
            Path(
                os.path.join(
                    "data",
                    "mura-inspect",
                    "negative",
                    str(i) + "_" + os.path.basename(img),
                )
            ),
        )
    else:
        copy(
            Path(img),
            Path(
                os.path.join(
                    "data", "mura-inspect", "positive", str(i) + "_" + os.path.basename(img) 
                )
            ),
        )
