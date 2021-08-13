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

# 训练
train_input_img_paths = sorted(
    [
        os.path.join(train_input_dir, fname)
        for fname in os.listdir(train_input_dir)
        if fname.endswith(".jpg")
    ]
)
train_target_img_paths = sorted(
    [
        os.path.join(val_target_dir, fname)
        for fname in os.listdir(val_target_dir)
        if fname.endswith(".png")
    ]
)
# 验证
val_input_img_paths = sorted(
    [
        os.path.join(val_input_dir, fname)
        for fname in os.listdir(val_input_dir)
        if fname.endswith(".jpg")
    ]
)
val_target_img_paths = sorted(
    [
        os.path.join(train_target_dir, fname)
        for fname in os.listdir(train_target_dir)
        if fname.endswith(".png")
    ]
)

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





from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class OxfordPets(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
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

        return x, y


from tensorflow.keras import layers


def Conv2D_BN(input,filter_number,kernel_size,padding='same',strides=1):
    input = layers.Conv2D(filter_number,kernel_size,strides=strides,activation=keras.activations.relu,padding=padding)(input)
    input = layers.BatchNormalization(axis=3)(input)
    return input
def inception(input,filter_number):
    b1 = Conv2D_BN(input,filter_number=filter_number,kernel_size=1,padding='same',strides=1)

    b3 = Conv2D_BN(input,filter_number=32,kernel_size=1,padding='same',strides=1)
    b3 = Conv2D_BN(b3,filter_number=filter_number,kernel_size=3,padding='same',strides=1)

    b5 = Conv2D_BN(input,filter_number=32,kernel_size=1,padding='same',strides=1)
    b5 = Conv2D_BN(b5,filter_number=filter_number,kernel_size=5,padding='same',strides=1)

    pool = layers.MaxPooling2D(pool_size=2,strides=1,padding='same')(input)
    pool = Conv2D_BN(pool,filter_number=filter_number,kernel_size=1,padding='same',strides=1)
    x = layers.Concatenate(axis=3)([b1,b3,b5,pool])

    return x



def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    # 160,160,3

    x = inception(inputs,32)
    # 160,160,3  ->  160,160,32
    temp1 = x
    x = layers.MaxPooling2D(2,strides=2,padding='same')(x)
    # 160,160,32 ->  80,80,32

    x = inception(x,64)
    # 80,80,32   ->  80,80,64
    temp2 = x
    x = layers.MaxPooling2D(2,strides=2,padding='same')(x)
    # 80,80,64   ->  40,40,64

    x = inception(x,128)
    # 40,40,64  ->  40,40,128
    temp3 = x
    x = layers.MaxPooling2D(2,strides=2,padding='same')(x)
    # 40,40,128  ->  20,20,128
    x = inception(x,256)
    # 20,20,128  ->  20,20,256
    temp4 = x


    #U-Net右半部分
    p4_up = layers.UpSampling2D(2)(temp4)
    # 20,20,256  ->  40,40,256
    p3 = layers.Concatenate(axis=3)([temp3,p4_up])
    # 40,40,256  +  40,40,128 = 40,40,384
    p3 = layers.Conv2D(128,3,strides=1,padding="same",activation=keras.activations.relu)(p3)
    p3 = layers.Conv2D(128,3,strides=1,padding="same",activation=keras.activations.relu)(p3)
    # 40,40,384  ->  40,40,128

    p3_up = layers.UpSampling2D(2)(p3)
    # 40,40,128  ->  80,80,128
    p2 = layers.Concatenate(axis=3)([temp2,p3_up])
    # 80,80,128  +  80,80,64  = 80,80,196
    p2 = layers.Conv2D(64,3,strides=1,padding="same",activation=keras.activations.relu)(p2)
    p2 = layers.Conv2D(64,3,strides=1,padding="same",activation=keras.activations.relu)(p2)
    # 80,80,196  ->  80,80,64

    p2_up = layers.UpSampling2D(2)(p2)
    # 80,80,64  ->  160,160,64
    p1 = layers.Concatenate(axis=3)([temp1,p2_up])
    # 160,160,64  + 160,160,32  =  160,160,96
    p1 = layers.Conv2D(32,3,strides=1,padding="same",activation=keras.activations.relu)(p1)
    p1 = layers.Conv2D(32,3,strides=1,padding="same",activation=keras.activations.relu)(p1)
    # 160,160,96  ->  160,160,32
    p1 = layers.Conv2D(num_classes,1,activation=keras.activations.softmax)(p1)
    # 160,160,32  ->  160,160,3

    model = keras.Model(inputs,p1)
    return model


keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()




train_gen = OxfordPets( batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen   = OxfordPets( batch_size, img_size, val_input_img_paths  , val_target_img_paths)
test_gen  = OxfordPets( batch_size, img_size, test_input_img_paths , test_target_img_paths)


model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",metrics=['accuracy'])


epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen)
model.save('oxford_GoogLeNet.h5')



