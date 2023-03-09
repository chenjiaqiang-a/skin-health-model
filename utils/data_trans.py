import random
import numpy as np
import torchvision.transforms.functional as tf
from torchvision import transforms

__all__ = ["BASIC_TRAIN_TRANS", "BASIC_TEST_TRANS",
           "image_density_trans_train", "image_density_trans_test"]

normalize = transforms.Normalize(mean=[0.5663, 0.4194, 0.3581],
                                 std=[0.3008,  0.2395, 0.2168])

BASIC_TRAIN_TRANS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    normalize
    ])
BASIC_TEST_TRANS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


def image_density_trans_train(img, density):
    top_ratio = random.random()
    left_ratio = random.random()
    flip_ratio = random.random()
    rotate_angle = random.randint(-10, 10)

    color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)

    img = tf.to_tensor(img)
    img = tf.resize(img, [560, 560])
    top = int((560 - 512) * top_ratio)
    left = int((560 - 512) * left_ratio)
    img = tf.crop(img, top, left, 512, 512)
    if flip_ratio < 0.5:
        img = tf.hflip(img)
    img = tf.rotate(img, rotate_angle)
    img = color_jitter(img)
    img = normalize(img)

    if density is None:
        density = np.zeros((140, 140))
    density = tf.to_tensor(density)
    density = tf.resize(density, [140, 140])
    top = int((140 - 128) * top_ratio)
    left = int((140 - 128) * left_ratio)
    density = tf.crop(density, top, left, 128, 128)
    if flip_ratio < 0.5:
        density = tf.hflip(density)
    density = tf.rotate(density, rotate_angle)

    return img, density


def image_density_trans_test(img, density):
    img = tf.to_tensor(img)
    img = tf.resize(img, [560, 560])
    img = tf.center_crop(img, [512, 512])
    img = normalize(img)

    if density is None:
        density = np.zeros((140, 140))
    density = tf.to_tensor(density)
    density = tf.resize(density, [140, 140])
    density = tf.center_crop(density, [128, 128])

    return img, density

