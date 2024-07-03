from utils import get_all_images
from pathlib import Path
import pandas as pd


PATH = Path("./data/Brain_AD/")

train_imgs = get_all_images(PATH / "test")
valid_imgs = get_all_images(PATH / "valid")

train_imgs = [img for img in train_imgs if "anomaly_mask" not in img]
train_imgs = [img for img in valid_imgs if "anomaly_mask" not in img]
# train
train_df = pd.DataFrame(
    {
        "path": [p.replace("data/", "") for p in train_imgs],
        "normal": [0 if "Ungood" in p else 1 for p in train_imgs],
    }
)
train_df.to_csv(PATH/ "train.csv", header=False, index=False)

# valid
valid_df = pd.DataFrame(
    {
        "path": [p.replace("data/", "") for p in valid_imgs],
        "normal": [0 if "Ungood" in p else 1 for p in valid_imgs],
    }
)
valid_df.to_csv(PATH / "valid.csv", header=False, index=False)
