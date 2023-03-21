import random
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

from utils.data_trans import image_density_train_trans

img_path = './data/images/CXM__何浩淼_痤疮毛囊炎_20190808083431000_斑点.jpg'
density_path = './data/density_maps/CXM__何浩淼_痤疮毛囊炎_20190808083431000_斑点.png'

image = Image.open(img_path)
density = Image.open(density_path)

top_ratio = random.random()
left_ratio = random.random()
flip_ratio = random.random()
rotate_angle = random.randint(-20, 20)

image = tf.resize(image, [256, 256])
top = int((256 - 224) * top_ratio)
left = int((256 - 224) * left_ratio)
image = tf.crop(image, top, left, 224, 224)
if flip_ratio < 0.5:
    image = tf.hflip(image)
image = tf.rotate(image, rotate_angle)

density = tf.resize(density, [128, 128])
top = int((128 - 112) * top_ratio)
left = int((128 - 112) * left_ratio)
density = tf.crop(density, top, left, 112, 112)
if flip_ratio < 0.5:
    density = tf.hflip(density)
density = tf.rotate(density, rotate_angle)

plt.subplot(1, 2, 1)
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.imshow(density)

plt.show()

