#!/usr/bin/env python
# coding: utf-8
import argparse
import datetime
import os

import numpy as np
import torch

import config
from models import DensityNet18
from models.loss_fn import get_loss_fn
from share import train_valid_split, get_optimizer
from utils import save_state_dict, save_result, Logger
from utils.data_trans import image_density_train_trans, image_density_test_trans
from utils.dataset import ImageWithDensity


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
    logger.info("Acne Severity Grading with Density Map: "
                f"{run_id} run by {config.EXP_RUNNER} on {device}")

    # Data Preparation
    train_dataset, valid_dataset, train_loader, valid_loader = train_valid_split(
        ImageWithDensity,
        config.TRAIN_CSV_PATH,
        config.IMAGE_DIR,
        args.valid_size,
        args.batch_size,
        train_args={
            "density_dir": config.DENSITY_MAP_DIR,
            "transform": image_density_train_trans,
        },
        valid_args={
            "density_dir": config.DENSITY_MAP_DIR,
            "transform": image_density_test_trans,
        }
    )
    logger.info(f"ImageWithDensity {len(train_dataset)} train samples "
                f"and {len(valid_dataset)} valid samples")

    # Model Preparation
    model = DensityNet18(config.NUM_CLASSES).to(device)
    logger.info("Using model DensityNet18")

    # Training Preparation
    loss_fn = get_loss_fn(args.loss_fn, reduction="mean")
    optimizer = get_optimizer(args.optim, model.parameters(), args.lr)
    logger.info("Training parameters:\n"
                f"batch_size      {args.batch_size}\n"
                f"epochs          {args.epochs}\n"
                f"loss_fn         {args.loss_fn}\n"
                f"learning_rate   {args.lr}\n"
                f"optimizer       {args.optim}")

    # Training
    result = train_and_valid(model, loss_fn, optimizer, train_loader, valid_loader,
                             args.epochs, device, model_folder, logger)

    # Saving Results
    save_state_dict(model, model_folder, "final-model.pth")
    save_result(result, run_folder, "result.pkl")
    logger.info(f"{run_id} is over!")


RESULT_ITEMS = ['loss_mse', 'loss', 'mae', 'mse', 'acc']


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
        train_ep_out = train_epoch(model, train_iter, loss_fn, optimizer, scheduler, device)
        valid_ep_out = valid_epoch(model, valid_iter, loss_fn, device)
        logger.info(f"Epoch {epoch:>3d}: "
                    f"TRAIN loss ({train_ep_out['loss_mse']:>8.3f},{train_ep_out['loss']:>8.5f}) "
                    f"metric ({train_ep_out['mae']:>8.3f},{train_ep_out['mse']:>8.3f},{train_ep_out['acc']:>6.3f}) | "
                    f"VALID loss ({valid_ep_out['loss_mse']:>8.3f},{valid_ep_out['loss']:>8.5f})"
                    f"metric ({valid_ep_out['mae']:>8.3f},{valid_ep_out['mse']:>8.3f},{valid_ep_out['acc']:>6.3f})")

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

    losses_mse = []
    losses = []
    total_density_sample = 0
    total_mae = 0
    total_mse = 0
    total_sample = 0
    total_correct = 0
    for images, (densities, d_masks), labels in data_iter:
        total_sample += len(labels)
        total_density_sample += torch.sum(torch.ones_like(d_masks)[d_masks]).numpy()
        images = images.to(device)
        densities, d_masks = densities.to(device), d_masks.to(device)
        labels = labels.to(device)

        density_out, out = model(images, densities, d_masks)
        loss_mse = torch.tensor(0, device=device)
        d_num = torch.sum(torch.ones_like(d_masks)[d_masks])
        if d_num > 0:
            loss_mse = torch.sum((density_out[d_masks] - densities[d_masks]) ** 2) / d_num
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        (loss_mse + loss).backward()
        optimizer.step()

        losses_mse.append(loss_mse.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())
        total_mae += torch.sum(torch.abs(density_out[d_masks] - densities[d_masks])).detach().cpu().numpy()
        total_mse += torch.sum((density_out[d_masks] - densities[d_masks]) ** 2).detach().cpu().numpy()
        total_correct += (torch.argmax(out, 1) == labels).int().sum().detach().cpu().numpy()
    scheduler.step()

    return {
        'loss_mse': np.mean(losses_mse),
        'loss': np.mean(losses),
        'mae': total_mae / total_density_sample,
        'mse': total_mse / total_density_sample,
        'acc': total_correct / total_sample,
    }


@torch.no_grad()
def valid_epoch(model, data_iter, loss_fn, device):
    model.eval()

    losses_mse = []
    losses = []
    total_density_sample = 0
    total_mae = 0
    total_mse = 0
    total_sample = 0
    total_correct = 0
    for images, (densities, d_masks), labels in data_iter:
        total_sample += len(labels)
        total_density_sample += torch.sum(torch.ones_like(d_masks)[d_masks]).numpy()
        images = images.to(device)
        densities, d_masks = densities.to(device), d_masks.to(device)
        labels = labels.to(device)

        density_out, out = model(images, densities, d_masks)
        loss_mse = torch.tensor(0, device=device)
        d_num = torch.sum(torch.ones_like(d_masks)[d_masks])
        if d_num > 0:
            loss_mse = torch.sum((density_out[d_masks] - densities[d_masks]) ** 2) / d_num
        loss = loss_fn(out, labels)

        losses_mse.append(loss_mse.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())
        total_mae += torch.sum(torch.abs(density_out[d_masks] - densities[d_masks])).detach().cpu().numpy()
        total_mse += torch.sum((density_out[d_masks] - densities[d_masks]) ** 2).detach().cpu().numpy()
        total_correct += (torch.argmax(out, 1) == labels).int().sum().detach().cpu().numpy()

    return {
        'loss_mse': np.mean(losses_mse),
        'loss': np.mean(losses),
        'mae': total_mae / total_density_sample,
        'mse': total_mse / total_density_sample,
        'acc': total_correct / total_sample,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_size', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_fn', type=str, choices=('ce', 'focal'), default='ce')
    parser.add_argument('--optim', type=str, choices=('adam', 'sgd'), default='adam')
    parser.add_argument('--run_folder', type=str, default='./run/density')

    arguments = parser.parse_args()
    main(arguments)
