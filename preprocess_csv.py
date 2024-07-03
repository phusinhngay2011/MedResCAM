from utils import get_all_images
from pathlib import Path
import pandas as pd


PATHS = [
    Path("./data/Brain_AD/"),
    Path("./data/Liver_AD/"),
    Path("./data/Retina_RESC_AD/"),
]

for path in PATHS:
    train_imgs = get_all_images(path / "test")
    valid_imgs = get_all_images(path / "valid")

    train_imgs = [img for img in train_imgs if "anomaly_mask" not in img]
    valid_imgs = [img for img in valid_imgs if "anomaly_mask" not in img]

    # train
    train_df = pd.DataFrame(
        {
            "path": [p.replace("data/", "") for p in train_imgs],
            "normal": [0 if "Ungood" in p else 1 for p in train_imgs],
        }
    )
    train_df.to_csv(path/ "train.csv", header=False, index=False)

    # valid
    valid_df = pd.DataFrame(
        {
            "path": [p.replace("data/", "") for p in valid_imgs],
            "normal": [0 if "Ungood" in p else 1 for p in valid_imgs],
        }
    )
    valid_df.to_csv(path / "valid.csv", header=False, index=False)
