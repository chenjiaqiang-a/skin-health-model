#!/usr/bin/env python
# coding: utf-8
import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models import ResNet50
from share import train_valid_split, get_optimizer
from utils import save_state_dict, save_result, Logger
from utils.data_trans import BASIC_TRAIN_TRANS, BASIC_TEST_TRANS
from utils.dataset import AcneDataset


class DistLoss(nn.Module):
    def __init__(self, score_1st=0.6, score_2nd=0.2, weight=None, reduction="mean") -> None:
        super(DistLoss, self).__init__()
        self.score_1st = score_1st
        self.score_2nd = score_2nd
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        hat = F.one_hot(torch.arange(inputs.shape[1])) * self.score_1st
        for i in range(1, inputs.shape[1]):
            hat[i, i - 1] = self.score_2nd
            hat[i - 1, i] = self.score_2nd
        hat = hat[targets]

        loss = -torch.log(torch.softmax(inputs, dim=1)) * hat.to(inputs.device)
        if self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "mean":
            return torch.mean(torch.sum(loss, dim=1))
        else:
            return torch.sum(loss, dim=1)

    def __str__(self):
        return "<DistLoss>"


def main(args):
    run_id = f'EXP-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_folder = os.path.join(args.run_folder, run_id)
    model_folder = os.path.join(run_folder, 'models')
    image_folder = os.path.join(run_folder, 'images')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        os.makedirs(model_folder)
        os.makedirs(image_folder)
    device = torch.device(config.DEVICE)
    logger = Logger(run_folder)
    logger.info(f"Acne Severity Grading by Dist Loss: {run_id} run by {config.EXP_RUNNER} on {device}")

    # Data Preparation
    train_dataset, valid_dataset, train_loader, valid_loader = train_valid_split(
        AcneDataset,
        config.TRAIN_CSV_PATH,
        config.IMAGE_DIR,
        args.valid_size,
        args.batch_size,
        train_args={"transform": BASIC_TRAIN_TRANS, "categories": config.ACNE_CATEGORIES},
        valid_args={"transform": BASIC_TEST_TRANS, "categories": config.ACNE_CATEGORIES}
    )
    logger.info(f"AcneDataset {len(train_dataset)} train samples "
                f"and {len(valid_dataset)} valid samples")

    # Model Preparation
    model = ResNet50(config.NUM_CLASSES).to(device)
    logger.info(f"Using model {model}")

    # Training Preparation
    loss_fn = DistLoss()
    optimizer = get_optimizer(args.optim, model.parameters(), args.lr)
    logger.info("Training parameters:\n"
                f"batch_size      {args.batch_size}\n"
                f"epochs          {args.epochs}\n"
                f"loss_fn         {loss_fn}\n"
                f"learning_rate   {args.lr}\n"
                f"optimizer       {args.optim}")

    # Training
    result = train_and_valid(model, loss_fn, optimizer, train_loader, valid_loader,
                             args.epochs, device, model_folder, logger)

    # Saving Results
    save_state_dict(model, model_folder, "final-model.pth")
    save_result(result, run_folder, "result.pkl")
    logger.info(f"{run_id} is over!")


RESULT_ITEMS = ['loss', 'acc']


def train_and_valid(model, loss_fn, optimizer, train_iter, valid_iter,
                    epochs, device, save_folder, logger):
    logger.info("Start Training...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    result = {
        'train': {item: [] for item in RESULT_ITEMS},
        'valid': {item: [] for item in RESULT_ITEMS}
    }
    best_valid_acc = 0
    for epoch in range(1, epochs + 1):
        if epoch % 4 == 0:
            loss_fn = DistLoss(1.2, -0.1)
        else:
            loss_fn = DistLoss(0.6, 0.2)
        train_ep_out = train_epoch(model, train_iter, loss_fn, optimizer, scheduler, device)
        valid_ep_out = valid_epoch(model, valid_iter, loss_fn, device)
        logger.info(f"Epoch {epoch:>3d}: "
                    f"TRAIN loss {train_ep_out['loss']:>7.5f} acc {train_ep_out['acc']:>5.3f} | "
                    f"VALID loss {valid_ep_out['loss']:>7.5f} acc {valid_ep_out['acc']:>5.3f}")

        for item in RESULT_ITEMS:
            result['train'][item].append(train_ep_out[item])
            result['valid'][item].append(valid_ep_out[item])
        if valid_ep_out['acc'] > best_valid_acc:
            best_valid_acc = valid_ep_out['acc']
            save_state_dict(model, save_folder, 'best-model.pth')
            logger.info("Saving Best...")

    return result


def train_epoch(model, data_iter, loss_fn, optimizer, scheduler, device):
    model.train()

    losses = []
    total_sample = 0
    total_correct = 0
    for images, labels in data_iter:
        total_sample += len(labels)
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        total_correct += (torch.argmax(output, 1) == labels).int().sum().detach().cpu().numpy()
    scheduler.step()

    return {
        'loss': np.mean(losses),
        'acc': total_correct / total_sample,
    }


@torch.no_grad()
def valid_epoch(model, data_iter, loss_fn, device):
    model.eval()

    losses = []
    total_correct = 0
    total_sample = 0
    for images, labels in data_iter:
        total_sample += len(labels)
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)

        losses.append(loss.detach().cpu().numpy())
        total_correct += (torch.argmax(output, 1) == labels).int().sum().detach().cpu().numpy()

    return {
        'loss': np.mean(losses),
        'acc': total_correct / total_sample,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_size', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, choices=('adam', 'sgd'), default='adam')
    parser.add_argument('--run_folder', type=str, default='./run/dist')

    arguments = parser.parse_args()
    main(arguments)
