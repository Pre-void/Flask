
import os


# 测试集
test_input_dir = "static/DataSet/test/images/"
test_target_dir = "static/DataSet/test/annotations/trimaps/"

img_size = (160, 160)
num_classes = 3
batch_size = 2

# 测试
test_input_img_paths = sorted(
    [
        os.path.join(test_input_dir, fname)
        for fname in os.listdir(test_input_dir)
        if fname.endswith(".jpg")
    ]
)
test_target_img_paths = sorted(
    [
        os.path.join(test_target_dir, fname)
        for fname in os.listdir(test_target_dir)
        if fname.endswith(".png")
    ]
)


import PIL
from PIL import ImageOps



"""
## Prepare `Sequence` class to load & vectorize batches of data
"""

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y




# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Res
model_Res = keras.models.load_model('oxford.h5')
# vgg
model_Vgg = keras.models.load_model('oxford_vgg.h5')
# GoogLeNet
model_GoogLeNet = keras.models.load_model('oxford_GoogLeNet.h5')




"""
## Visualize predictions
"""


test_gen  = OxfordPets( batch_size, img_size, test_input_img_paths , test_target_img_paths)
#
# ResNet
val_preds_Res = model_Res.predict(test_gen)
loss_R,acc_R  = model_Res.evaluate(test_gen)
print(str(loss_R)+'----'+str(acc_R))
# VGG
val_preds_Vgg = model_Vgg.predict(test_gen)
loss_V,acc_V  = model_Vgg.evaluate(test_gen)
print(str(loss_V)+'----'+str(acc_V))

# GoogLeNet
#[n,160,160,3]
val_preds_GoogLe = model_GoogLeNet.predict(test_gen)
loss_G,acc_G  = model_GoogLeNet.evaluate(test_gen)
print(str(loss_G)+'----'+str(acc_G))


loss = loss_V
acc  = acc_V

def display_mask(i,type):
    if type=='VGG':
        # [160,160,3]
        mask = np.argmax(val_preds_Vgg[i], axis=-1)
        # [160,160]
    elif type=='ResNet':
        mask = np.argmax(val_preds_Res[i], axis=-1)
    elif type=='GoogLeNet':
        mask = np.argmax(val_preds_GoogLe[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    #     [160,160,1]
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    # display(img)
    return img





import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
i = random.randint(0,len(test_input_img_paths)-1)
def get_result(i,type):
    img1  = mpimg.imread(test_input_img_paths[i])
    img1_ = mpimg.imread(test_target_img_paths[i])
    img1__ = display_mask(i,type)
    if type=='VGG':
        loss = loss_V
        acc = acc_V
        title = 'VGG  loss:' + str(loss)[0:5] + '  acc:' + str(acc)[0:5]
        fname = 'static/result/' + str(i) + '_vgg.jpg'
    elif type=='ResNet':
        loss = loss_R
        acc = acc_R
        title = 'ResNet  loss:' + str(loss)[0:5] + '  acc:' + str(acc)[0:5]
        fname = 'static/result/' + str(i) + '_res.jpg'
    elif type=='GoogLeNet':
        loss = loss_G
        acc = acc_G
        title = 'GoogLeNet  loss:' + str(loss)[0:5] + '  acc:' + str(acc)[0:5]
        fname = 'static/result/' + str(i) + '_google.jpg'

    # plt.figure("Image") # 图像窗口名称


    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Input')

    plt.subplot(1,3,2)
    plt.imshow(img1_)
    plt.axis('off')
    plt.title('Ground Truth')

    plt.subplot(1,3,3)
    plt.imshow(img1__)
    plt.axis('off')
    plt.title('Predict')


    plt.suptitle(title, y=0.1)
    plt.savefig(fname=fname,figsize=[10,10])
