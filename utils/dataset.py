import os
from collections import Counter

import pandas as pd
from torch.utils import data
from PIL import Image

__all__ = ['AcneDataset', 'ImageWithDensity']


class AcneDataset(data.Dataset):
    def __init__(self, csv_file, image_dir, categories=None, transform=None, **kwargs):
        super(AcneDataset, self).__init__(**kwargs)
        self.image_dir = image_dir

        self.df = pd.read_csv(csv_file)
        self.image_files = list(self.df['filename'])
        self.labels = list(self.df['label'])

        self.counter = Counter(self.labels)
        if categories is None:
            categories = [i for i in self.counter]
        self.categories = categories
        self._category_to_label = {
            category: label for label, category in enumerate(self.categories)
        }

        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.image_files)

    def to_labels(self, category_list):
        return [self._category_to_label[c] for c in category_list]

    def to_categories(self, label_list):
        return [self.categories[label] for label in label_list]


class ImageWithDensity(AcneDataset):
    def __init__(self, csv_file, image_dir, density_dir, categories=None, transform=None, **kwargs):
        super(ImageWithDensity, self).__init__(csv_file, image_dir, categories, transform, **kwargs)
        self.density_dir = density_dir
        self.density_files = list(self.df['density_map'].fillna(0))

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path)

        density = None
        d_mask = False
        if self.density_files[index]:
            density_path = os.path.join(self.density_dir, self.density_files[index])
            density = Image.open(density_path)
            d_mask = True

        if self.transform is not None:
            image, density = self.transform(image, density)

        label = self.labels[index]

        return image, (density, d_mask), label
