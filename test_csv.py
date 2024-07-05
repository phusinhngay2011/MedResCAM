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

# images = get_all_images("./data/mura")

# for img in tqdm(images):
#     if "negative" in img:
#         copy(Path(img), Path(os.path.join("data", "mura-inspect", "negative", os.path.basename(img))))
#     else:
#         copy(Path(img), Path(os.path.join("data", "mura-inspect", "positive", os.path.basename(img))))

import deeplake

ds = deeplake.load("hub://activeloop/mura-train")
dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
dataloader = ds.tensorflow()
