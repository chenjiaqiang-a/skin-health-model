import random
import numpy as np
import torchvision.transforms.functional as tf
from torchvision import transforms

__all__ = ["BASIC_TRAIN_TRANS", "BASIC_TEST_TRANS",
           "image_density_train_trans", "image_density_test_trans"]

normalize = transforms.Normalize(mean=[0.5663, 0.4194, 0.3581],
                                 std=[0.3008,  0.2395, 0.2168])

BASIC_TRAIN_TRANS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalize
    ])
BASIC_TEST_TRANS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def image_density_train_trans(image, density):
    top_ratio = random.random()
    left_ratio = random.random()
    flip_ratio = random.random()
    rotate_angle = random.randint(-10, 10)

    image = tf.resize(image, [256, 256])
    top = int((256 - 224) * top_ratio)
    left = int((256 - 224) * left_ratio)
    image = tf.crop(image, top, left, 224, 224)
    if flip_ratio < 0.5:
        image = tf.hflip(image)
    image = tf.rotate(image, rotate_angle)
    image = tf.to_tensor(image)
    image = normalize(image)

    if density is None:
        density = np.zeros((128, 128), dtype=np.float32)
    density = tf.to_tensor(density)
    density = tf.resize(density, [128, 128])
    top = int((128 - 112) * top_ratio)
    left = int((128 - 112) * left_ratio)
    density = tf.crop(density, top, left, 112, 112)
    if flip_ratio < 0.5:
        density = tf.hflip(density)
    density = tf.rotate(density, rotate_angle)

    return image, density


def image_density_test_trans(image, density):
    image = tf.resize(image, [256, 256])
    image = tf.center_crop(image, [224, 224])
    image = tf.to_tensor(image)
    image = normalize(image)

    if density is None:
        density = np.zeros((128, 128), dtype=np.float32)
    density = tf.to_tensor(density)
    density = tf.resize(density, [128, 128])
    density = tf.center_crop(density, [112, 112])

    return image, density

