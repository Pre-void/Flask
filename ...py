# 训练集
train_input_dir = "static/DataSet/train/images/"
train_target_dir = "static/DataSet/train/annotations/trimaps/"
# 验证集
val_input_dir = "static/DataSet/val/images/"
val_target_dir = "static/DataSet/val/annotations/trimaps/"
# 测试集
test_input_dir = "static/DataSet/test/images/"
test_target_dir = "static/DataSet/val/annotations/trimaps/"




img_size = (160, 160)
num_classes = 3
batch_size = 32

import os

val_target_img_paths = sorted(
    [
        os.path.join(train_target_dir, fname)
        for fname in os.listdir(train_target_dir)
        if fname.endswith(".png")
    ]
)

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
path = val_target_img_paths[0]

img = load_img(path, target_size=img_size, color_mode="grayscale")
y = np.zeros((batch_size,) + img_size + (1,), dtype="uint8")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.imshow(y[0])
plt.show()