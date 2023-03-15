import os
from collections import Counter

import pandas as pd
from torch.utils import data
from PIL import Image

__all__ = ['ACNE_CATEGORIES', 'AcneDataset', 'ImageWithMultiLabel',
           'ImageWithDensity', 'ImageWithDensityAndMultiLabel']

ACNE_CATEGORIES_True = ['Clear', 'Almost', 'Mild',
                        'Mild to Moderate', 'Moderate',
                        'Moderate to Less Severe',
                        'Less Severe', 'Severe']
ACNE_CATEGORIES = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']


class AcneDataset(data.Dataset):
    def __init__(self, csv_file, image_dir, categories=None, transform=None, **kwargs):
        super(AcneDataset, self).__init__(**kwargs)
        self.image_dir = image_dir

        self.df = pd.read_csv(csv_file)
        self.image_files = list(self.df['filename'])
        self.labels = list(self.df['label'])

        if categories is None:
            categories = ACNE_CATEGORIES
        self.categories = categories
        self.counter = Counter(self.labels)
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


LEVEL_1ST_LABEL_MAPS = {
    'base': [0, 0, 1, 0, 0, 0, 0, 0]
}
LEVEL_2ND_LABEL_MAPS = {
    'base': [0, 0, 1, 2, 3, 0, 3, 2]
}


class ImageWithMultiLabel(AcneDataset):
    def __init__(self, csv_file, image_dir,
                 label_map_1st='base', label_map_2nd='base',
                 categories=None, transform=None, **kwargs):
        super(ImageWithMultiLabel, self).__init__(csv_file, image_dir, categories, transform, **kwargs)
        self.label_map_1st = LEVEL_1ST_LABEL_MAPS[label_map_1st]
        self.label_map_2nd = LEVEL_2ND_LABEL_MAPS[label_map_2nd]

    def __getitem__(self, index):
        image, label = super(ImageWithMultiLabel, self).__getitem__(index)
        label_1st = self.label_map_1st[label]
        label_2nd = self.label_map_2nd[label]

        return image, (label_1st, label_2nd, label)


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


class ImageWithDensityAndMultiLabel(ImageWithDensity):
    def __init__(self, csv_file, image_dir, density_dir,
                 label_map_1st='base', label_map_2nd='base',
                 categories=None, transform=None, **kwargs):
        super(ImageWithDensityAndMultiLabel, self).__init__(csv_file, image_dir, density_dir,
                                                            categories, transform, **kwargs)
        self.label_map_1st = LEVEL_1ST_LABEL_MAPS[label_map_1st]
        self.label_map_2nd = LEVEL_2ND_LABEL_MAPS[label_map_2nd]

    def __getitem__(self, index):
        image, (density, d_mask), label = super(ImageWithDensityAndMultiLabel, self).__getitem__(index)
        label_1st = self.label_map_1st[label]
        label_2nd = self.label_map_2nd[label]

        return image, (density, d_mask), (label_1st, label_2nd, label)
