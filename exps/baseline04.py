import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
from torchvision import transforms

import config
from utils import save_state_dict

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device(config.DEVICE)
RUN_FOLDER = './run/exps/0004'
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)


class AcneDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        super(AcneDataset, self).__init__()
        self.image_dir = image_dir

        df = pd.read_csv(csv_file)
        self.image_files = list(df['filename'])
        self.labels = list(df['label'])

        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]

        return image, label

    def __len__(self):
        return len(self.image_files)


class BaselineModel(nn.Module):
    def __init__(self, out_dims: int) -> None:
        super(BaselineModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, out_dims)

    def forward(self, images):
        return self.resnet(images)


# Data Preparation
normalize = transforms.Normalize(mean=[0.5663, 0.4194, 0.3581],
                                 std=[0.3008, 0.2395, 0.2168])
train_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    normalize,
])
test_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
train_dataset = AcneDataset(config.TRAIN_CSV_PATH,
                            config.IMAGE_DIR,
                            transform=train_trans)
test_dataset = AcneDataset(config.TEST_CSV_PATH,
                           config.IMAGE_DIR,
                           transform=test_trans)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

# Model Preparation
model = BaselineModel(config.NUM_CLASSES).to(DEVICE)


# loss_fn = nn.CrossEntropyLoss()
def loss_fn(outputs, targets, reduction="mean"):
    hat = F.one_hot(torch.arange(outputs.shape[1])) * 0.8
    for i in range(1, outputs.shape[1]):
        hat[i, i - 1] = 0.1
        hat[i - 1, i] = 0.1
    hat = hat[targets]

    loss = -torch.log(torch.softmax(outputs, dim=1)) * hat.to(outputs.device)
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(torch.sum(loss, dim=1))
    else:
        return torch.sum(loss, dim=1)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

result = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
}
best_acc = 0
for ep in range(EPOCHS):
    model.train()
    losses = []
    total_sample = 0
    total_correct = 0
    for images, labels in train_loader:
        total_sample += len(labels)
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        total_correct += (torch.argmax(outputs, 1) == labels).int().sum().detach().cpu().numpy()
    result['train_loss'].append(np.mean(losses))
    result['train_acc'].append(total_correct / total_sample)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_sample = 0
        losses = []
        for images, labels in test_loader:
            total_sample += len(labels)
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            losses.append(loss.cpu().numpy())
            total_correct += (torch.argmax(outputs, 1) == labels).int().sum().cpu().numpy()
        result['test_loss'].append(np.mean(losses))
        result['test_acc'].append(total_correct / total_sample)

    print(f"EPOCH {ep:>03d} TRAIN loss {result['train_loss'][-1]:>6.4f} acc {result['train_acc'][-1]:>5.3f} "
          f"TEST loss {result['test_loss'][-1]:>6.4f} acc {result['test_acc'][-1]:>5.3f}")
    if result['test_acc'][-1] > best_acc:
        best_acc = result['test_acc'][-1]
        save_state_dict(model, RUN_FOLDER, 'best-model.pth')

    df = pd.DataFrame.from_dict(result)
    df.to_csv(os.path.join(RUN_FOLDER, 'result.csv'))
