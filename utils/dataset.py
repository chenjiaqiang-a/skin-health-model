import os
from collections import Counter

import pandas as pd
from torch.utils import data
from PIL import Image

__all__ = ['ACNE_CATEGORIES', 'AcneImageDataset']

ACNE_CATEGORIES = ['Clear', 'Almost', 'Mild',
                   'Mild to Moderate', 'Moderate',
                   'Moderate to Less Severe',
                   'Less Severe', 'Severe']


class AcneImageDataset(data.Dataset):
    def __init__(self, csv_file_dir, file_dir,
                 categories=None,
                 transform=None, **kwargs):
        super(AcneImageDataset, self).__init__(**kwargs)
        self.file_dir = file_dir

        df = pd.read_csv(csv_file_dir)
        self.filenames = list(df['filename'])
        self.labels = list(df['label'])

        if categories is None:
            categories = ACNE_CATEGORIES
        self.categories = categories
        self.counter = Counter(self.labels)
        if len(self.categories) != len(self.counter):
            raise ValueError(f'The number of categories({len(self.categories)}) should equal to the '
                             f'number of labels\' kinds({len(self.counter)})!')
        self._category_to_label = {
            category: label for label, category in enumerate(self.categories)
        }

        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.file_dir, self.filenames[index])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.filenames)

    def to_labels(self, category_list):
        return [self._category_to_label[c] for c in category_list]

    def to_categories(self, label_list):
        return [self.categories[l] for l in label_list]
