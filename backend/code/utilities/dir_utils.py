import os
from datetime import datetime


def make_and_get_images_dir():
    img_dir = f"./images/{datetime.today().strftime('%Y/%m/%d')}"
    os.makedirs(img_dir, exist_ok=True)
    return img_dir
