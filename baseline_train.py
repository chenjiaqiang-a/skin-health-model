#!/usr/bin/env python
# coding: utf-8
# 基础模型训练
# 指定gpu请设置环境变量 CUDA_VISIBLE_DEVICES
import os
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models.baseline import ResNet50Baseline
from models.loss_fn import FocalLoss
from utils.dataset import AcneImageDataset
from utils.data_trans import BASIC_TRAIN_TRANS
from utils import save_state_dict, save_result, Logger


def main(args):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = os.path.join(args.run_folder, run_id)
    model_folder = os.path.join(run_folder, 'models')
    image_folder = os.path.join(run_folder, 'images')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        os.makedirs(model_folder)
        os.makedirs(image_folder)
    logger = Logger(run_folder, "train")
    device = torch.device('cuda:0')
    logger.info(f"Ex({run_id}) run by Chen: Baseline train on {device}")

    # Data Preparation
    dataset = AcneImageDataset('./data/HX_Acne_Image_GroundTruth_Train.csv',
                               './data/images',
                               transform=BASIC_TRAIN_TRANS)
    valid_size = int(len(dataset) * args.val_size)
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    os.remove(os.path.join(run_folder, 'temp_train.csv'))
    os.remove(os.path.join(run_folder, 'temp_valid.csv'))
    logger.info(f"Train on HX Skin Dataset(Acne) with {len(train_dataset)} train samples, "
                f"{len(valid_dataset)} valid samples")

    # Model Preparation
    model = ResNet50Baseline().to(device)

    # Training Preparation
    if args.loss == 'focal':
        criterion = FocalLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss(reduction="none")
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    # Train
    logger.info("Training parameters:")
    logger.info(f"batch_size      {args.batch_size}")
    logger.info(f"epochs          {args.epochs}")
    logger.info(f"learning_rate   {args.lr}")
    logger.info(f"optimizer       {args.opt}")
    logger.info(f"early_threshold {args.early_threshold}")
    logger.info(f"val_size        {args.val_size}")
    logger.info(f"model           ResNet50Baseline")
    logger.info(f"criterion       {args.loss}")

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_valid_acc = 0
    epoch_counter = args.early_threshold
    logger.info("Start Training...")
    for epoch in range(1, args.epochs + 1):
        train_output = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_loss.append(train_output['loss'])
        train_acc.append(train_output['acc'])

        valid_output = valid_epoch(model, valid_loader, criterion, device)
        valid_loss.append(valid_output['loss'])
        valid_acc.append(valid_output['acc'])

        logger.info(f"Epoch {epoch:>3d}: "
                    f"TRAIN loss {train_loss[-1]:>10.6f} acc {train_acc[-1]:>6.4f} | "
                    f"VALID loss {valid_loss[-1]:>10.6f} acc {valid_acc[-1]:>6.4f}")

        if valid_output['acc'] > best_valid_acc:
            best_valid_acc = valid_output['acc']
            epoch_counter = args.early_threshold
            save_state_dict(model, model_folder, "best-model.pth")
            logger.info("Saving Best...")
        else:
            epoch_counter -= 1

        if epoch_counter == 0:
            logger.info('Early Stopped!')
            break

    save_state_dict(model, model_folder, "final-model.pth")
    save_result({
        "train_param": f"{args.loss}-base",
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
    }, run_folder, "result.pkl")
    logger.info(f"Ex({run_id}) is over!")


def train_epoch(model, data_iter, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_sample = 0
    for images, labels in data_iter:
        total_sample += len(labels)
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)
        total_loss += loss.sum().detach().cpu().item()

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        prediction = torch.argmax(output, 1)
        correct = (prediction == labels).sum().int().detach().cpu().item()
        total_correct += correct
    scheduler.step()

    epoch_loss = total_loss / total_sample
    epoch_acc = total_correct / total_sample
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
    }


def valid_epoch(model, data_iter, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_sample = 0

    with torch.no_grad():
        for images, labels in data_iter:
            total_sample += len(labels)
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_fn(output, labels)
            total_loss += loss.sum().cpu().item()

            prediction = torch.argmax(output, 1)
            correct = (prediction == labels).sum().int().cpu().item()
            total_correct += correct

    epoch_loss = total_loss / total_sample
    epoch_acc = total_correct / total_sample
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--run_folder', type=str, default='./run/baseline')
    parser.add_argument('--early_threshold', type=int, default=20)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--loss', type=str, default='ce', choices=('ce', 'focal'))
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))

    arguments = parser.parse_args()
    main(arguments)
