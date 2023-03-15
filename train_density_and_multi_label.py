#!/usr/bin/env python
# coding: utf-8
import argparse
import datetime
import os

import numpy as np
import torch

import config
from models import DensityWithMultiLabelNet18
from models.loss_fn import get_loss_fn
from share import train_valid_split, get_optimizer
from utils import save_state_dict, save_result, Logger
from utils.data_trans import image_density_train_trans, image_density_test_trans
from utils.dataset import ImageWithDensityAndMultiLabel


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
    logger.info("Acne Severity Grading with Density Map and Multi-level Category Labels: "
                f"{run_id} run by {config.EXP_RUNNER} on {device}")

    # Data Preparation
    train_dataset, valid_dataset, train_loader, valid_loader = train_valid_split(
        ImageWithDensityAndMultiLabel,
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
    logger.info(f"ImageWithDensityAndMultiLabel {len(train_dataset)} train samples "
                f"and {len(valid_dataset)} valid samples")

    # Model Preparation
    model = DensityWithMultiLabelNet18(config.NUM_1ST_LEVEL_CLASSES,
                                       config.NUM_2ND_LEVEL_CLASSES,
                                       config.NUM_CLASSES).to(device)
    logger.info("Using model DensityWithMultiLabelNet18 with"
                f"label_1st_map={train_dataset.label_map_1st}, "
                f"label_map_2nd={train_dataset.label_map_2nd}")

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


RESULT_ITEMS = ['loss_mse', 'loss_1st', 'loss_2nd', 'loss',
                'mae', 'mse', 'acc_1st', 'acc_2nd', 'acc']


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
                    f"TRAIN loss ({train_ep_out['loss_mse']:>8.3f},{train_ep_out['loss_1st']:>8.5f},"
                    f"{train_ep_out['loss_2nd']:>8.5f},{train_ep_out['loss']:>8.5f}) "
                    f"metric ({train_ep_out['mae']:>8.3f},{train_ep_out['mse']:>8.3f},{train_ep_out['acc_1st']:>6.3f},"
                    f"{train_ep_out['acc_2nd']:>6.3f},{train_ep_out['acc']:>6.3f}) | "
                    f"VALID loss ({valid_ep_out['loss_mse']:>8.3f},{valid_ep_out['loss_1st']:>8.5f},"
                    f"{valid_ep_out['loss_2nd']:>8.5f},{valid_ep_out['loss']:>8.5f}) "
                    f"metric ({valid_ep_out['mae']:>8.3f},{valid_ep_out['mse']:>8.3f},{valid_ep_out['acc_1st']:>6.3f},"
                    f"{valid_ep_out['acc_2nd']:>6.3f},{valid_ep_out['acc']:>6.3f})")

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
    losses_1st = []
    losses_2nd = []
    losses = []
    total_density_sample = 0
    total_mae = 0
    total_mse = 0
    total_sample = 0
    correct_1st = 0
    correct_2nd = 0
    correct = 0
    for images, (densities, d_masks), (labels_1st, labels_2nd, labels) in data_iter:
        total_sample += len(labels)
        total_density_sample += torch.sum(torch.ones_like(d_masks)[d_masks]).numpy()
        images = images.to(device)
        densities, d_masks = densities.to(device), d_masks.to(device)
        labels_1st, labels_2nd, labels = labels_1st.to(device), labels_2nd.to(device), labels.to(device)

        density_out, (out_1st, out_2nd, out) = model(images, densities, d_masks)
        loss_mse = torch.tensor(0, device=device)
        d_num = torch.sum(torch.ones_like(d_masks)[d_masks])
        if d_num > 0:
            loss_mse = torch.sum((density_out[d_masks] - densities[d_masks]) ** 2) / d_num
        loss_1st = loss_fn(out_1st, labels_1st)
        loss_2nd = loss_fn(out_2nd, labels_2nd)
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        (loss_mse + loss + loss_1st + loss_2nd).backward()
        optimizer.step()

        losses_mse.append(loss_mse.detach().cpu().numpy())
        losses_1st.append(loss_1st.detach().cpu().numpy())
        losses_2nd.append(loss_2nd.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())
        total_mae += torch.sum(torch.abs(density_out[d_masks] - densities[d_masks])).detach().cpu().numpy()
        total_mse += torch.sum((density_out[d_masks] - densities[d_masks])**2).detach().cpu().numpy()
        correct_1st += (torch.argmax(out_1st, 1) == labels_1st).int().sum().detach().cpu().numpy()
        correct_2nd += (torch.argmax(out_2nd, 1) == labels_2nd).int().sum().detach().cpu().numpy()
        correct += (torch.argmax(out, 1) == labels).int().sum().detach().cpu().numpy()
    scheduler.step()

    return {
        'loss_mse': np.mean(losses_mse),
        'loss_1st': np.mean(losses_1st),
        'loss_2nd': np.mean(losses_2nd),
        'loss': np.mean(losses),
        'mae': total_mae / total_density_sample,
        'mse': total_mse / total_density_sample,
        'acc_1st': correct_1st / total_sample,
        'acc_2nd': correct_2nd / total_sample,
        'acc': correct / total_sample,
    }


@torch.no_grad()
def valid_epoch(model, data_iter, loss_fn, device):
    model.eval()

    losses_mse = []
    losses_1st = []
    losses_2nd = []
    losses = []
    total_density_sample = 0
    total_mae = 0
    total_mse = 0
    total_sample = 0
    correct_1st = 0
    correct_2nd = 0
    correct = 0
    for images, (densities, d_masks), (labels_1st, labels_2nd, labels) in data_iter:
        total_sample += len(labels)
        total_density_sample += torch.sum(torch.ones_like(d_masks)[d_masks]).numpy()
        images = images.to(device)
        densities, d_masks = densities.to(device), d_masks.to(device)
        labels_1st, labels_2nd, labels = labels_1st.to(device), labels_2nd.to(device), labels.to(device)

        density_out, (out_1st, out_2nd, out) = model(images, densities, d_masks)
        loss_mse = torch.tensor(0, device=device)
        d_num = torch.sum(torch.ones_like(d_masks)[d_masks])
        if d_num > 0:
            loss_mse = torch.sum((density_out[d_masks] - densities[d_masks]) ** 2) / d_num
        loss_1st = loss_fn(out_1st, labels_1st)
        loss_2nd = loss_fn(out_2nd, labels_2nd)
        loss = loss_fn(out, labels)

        losses_mse.append(loss_mse.detach().cpu().numpy())
        losses_1st.append(loss_1st.detach().cpu().numpy())
        losses_2nd.append(loss_2nd.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())
        total_mae += torch.sum(torch.abs(density_out[d_masks] - densities[d_masks])).detach().cpu().numpy()
        total_mse += torch.sum((density_out[d_masks] - densities[d_masks]) ** 2).detach().cpu().numpy()
        correct_1st += (torch.argmax(out_1st, 1) == labels_1st).int().sum().detach().cpu().numpy()
        correct_2nd += (torch.argmax(out_2nd, 1) == labels_2nd).int().sum().detach().cpu().numpy()
        correct += (torch.argmax(out, 1) == labels).int().sum().detach().cpu().numpy()

    return {
        'loss_mse': np.mean(losses_mse),
        'loss_1st': np.mean(losses_1st),
        'loss_2nd': np.mean(losses_2nd),
        'loss': np.mean(losses),
        'mae': total_mae / total_density_sample,
        'mse': total_mse / total_density_sample,
        'acc_1st': correct_1st / total_sample,
        'acc_2nd': correct_2nd / total_sample,
        'acc': correct / total_sample,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_size', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_fn', type=str, choices=('ce', 'focal'), default='ce')
    parser.add_argument('--optim', type=str, choices=('adam', 'sgd'), default='adam')
    parser.add_argument('--run_folder', type=str, default='./run/density_and_multi_label')

    arguments = parser.parse_args()
    main(arguments)
