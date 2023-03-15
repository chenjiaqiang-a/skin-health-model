import os
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


def get_optimizer(opt, parameters, lr):
    if opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    return optimizer


def train_valid_split(
        dataset,
        csv_file: str,
        image_dir: str,
        valid_size: float,
        batch_size: int,
        train_args: dict,
        valid_args: dict):
    df = pd.read_csv(csv_file)
    train_x, valid_x, train_y, valid_y = train_test_split(df, df['label'], test_size=valid_size, random_state=10)
    train_x.to_csv('temp-train.csv', index=False)
    valid_x.to_csv('temp-valid.csv', index=False)

    train_dataset = dataset('temp-train.csv', image_dir=image_dir, **train_args)
    valid_dataset = dataset('temp-valid.csv', image_dir=image_dir, **valid_args)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=False, num_workers=4)

    os.remove('temp-train.csv')
    os.remove('temp.valid.csv')
    return train_dataset, valid_dataset, train_loader, valid_loader
