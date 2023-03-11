import os
import datetime
import argparse
from copy import copy

import torch
import numpy as np

import config
from share import train_valid_split, get_optimizer
from models.networks import ACCNet
from models.loss_fn import get_loss_fn
from utils.dataset import ImageWithDensityAnd3LevelLabel
from utils.data_trans import image_density_trans_train, image_density_trans_test
from utils import Logger, save_state_dict, save_result


def main(args):
    # Environment Setting
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
    dataset = ImageWithDensityAnd3LevelLabel(config.TRAIN_CSV_PATH,
                                             config.IMAGE_DIR,
                                             config.DENSITY_MAP_DIR)
    train_loader, valid_loader = train_valid_split(dataset, args.valid_size, args.batch_size)
    train_loader.dataset = copy(dataset)
    train_loader.dataset.transform = image_density_trans_train
    valid_loader.dataset.transform = image_density_trans_test
    logger.info(f"Total number of samples: {len(dataset)}"
                f"({len(dataset) - int(len(dataset) * args.valid_size)} for train, "
                f"{int(len(dataset) * args.valid_size)} for valid)")

    # Model Preparation
    model = ACCNet(config.NUM_CLASSES, config.NUM_1ST_LEVEL_CLASSES, config.NUM_2ND_LEVEL_CLASSES).to(device)

    # Training Preparation
    loss_fn = get_loss_fn(args.loss_fn, reduction='mean')
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
    save_state_dict(model, model_folder, 'final-model.pth')
    save_result(result, run_folder)


def train_and_valid(model, loss_fn, optimizer,
                    train_loader, valid_loader,
                    epochs, device, save_folder, logger):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    train_out = {
        'losses': {
            'loss_mse': [],
            'loss_1st': [],
            'loss_2nd': [],
            'loss': [],
        },
        'metric': {
            'mae': [],
            'mse': [],
            'acc_1st': [],
            'acc_2nd': [],
            'acc': [],
        }
    }
    valid_out = {
        'losses': {
            'loss_mse': [],
            'loss_1st': [],
            'loss_2nd': [],
            'loss': [],
        },
        'metric': {
            'mae': [],
            'mse': [],
            'acc_1st': [],
            'acc_2nd': [],
            'acc': [],
        }
    }
    best_valid_acc = 0
    logger.info(f"Epoch 000: TRAIN losses {'mse loss':>10s} "
                f"{'1st loss':>10s} {'2nd loss':>10s} {'loss':>10s} "
                f"metric {'mae':>10s} {'mse':>10s} {'acc1':>4s} {'acc2':>4s} {'acc':>4s} | "
                f"VALID losses {'mse loss':>10s} "
                f"{'1st loss':>10s} {'2nd loss':>10s} {'loss':>10s} "
                f"metric {'mae':>10s} {'mse':>10s} {'acc1':>4s} {'acc2':>4s} {'acc':>4s}")
    for epoch in range(1, epochs + 1):
        train_ep_out = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        valid_ep_out = valid_epoch(model, valid_loader, loss_fn, device)
        logger.info(f"Epoch {epoch:>3d}: "
                    f"TRAIN losses {np.mean(train_ep_out['losses']['loss_mse']):>10.6f} "
                    f"{np.mean(train_ep_out['losses']['loss_1st']):>10.6f} "
                    f"{np.mean(train_ep_out['losses']['loss_2nd']):>10.6f} "
                    f"{np.mean(train_ep_out['losses']['loss']):>10.6f} "
                    f"metric {train_ep_out['metric']['mae']:>10.6f} "
                    f"{train_ep_out['metric']['mse']:>10.6f} "
                    f"{train_ep_out['metric']['acc_1st']:>4.3f} "
                    f"{train_ep_out['metric']['acc_2nd']:>4.3f} "
                    f"{train_ep_out['metric']['acc']:>4.3f} | "
                    f"VALID losses {np.mean(valid_ep_out['losses']['loss_mse']):>10.6f} "
                    f"{np.mean(valid_ep_out['losses']['loss_1st']):>10.6f} "
                    f"{np.mean(valid_ep_out['losses']['loss_2nd']):>10.6f} "
                    f"{np.mean(valid_ep_out['losses']['loss']):>10.6f} "
                    f"metric {valid_ep_out['metric']['mae']:>10.6f} "
                    f"{valid_ep_out['metric']['mse']:>10.6f} "
                    f"{valid_ep_out['metric']['acc_1st']:>4.3f} "
                    f"{valid_ep_out['metric']['acc_2nd']:>4.3f} "
                    f"{valid_ep_out['metric']['acc']:>4.3f}")

        train_out['losses']['loss_mse'].extend(train_ep_out['losses']['loss_mse'])
        train_out['losses']['loss_1st'].extend(train_ep_out['losses']['loss_1st'])
        train_out['losses']['loss_2nd'].extend(train_ep_out['losses']['loss_2nd'])
        train_out['losses']['loss'].extend(train_ep_out['losses']['loss'])
        train_out['metric']['mae'].append(train_ep_out['metric']['mae'])
        train_out['metric']['mse'].append(train_ep_out['metric']['mse'])
        train_out['metric']['acc_1st'].append(train_ep_out['metric']['acc_1st'])
        train_out['metric']['acc_2nd'].append(train_ep_out['metric']['acc_2nd'])
        train_out['metric']['acc'].append(train_ep_out['metric']['acc'])
        valid_out['losses']['loss_mse'].extend(valid_ep_out['losses']['loss_mse'])
        valid_out['losses']['loss_1st'].extend(valid_ep_out['losses']['loss_1st'])
        valid_out['losses']['loss_2nd'].extend(valid_ep_out['losses']['loss_2nd'])
        valid_out['losses']['loss'].extend(valid_ep_out['losses']['loss'])
        valid_out['metric']['mae'].append(valid_ep_out['metric']['mae'])
        valid_out['metric']['mse'].append(valid_ep_out['metric']['mse'])
        valid_out['metric']['acc_1st'].append(valid_ep_out['metric']['acc_1st'])
        valid_out['metric']['acc_2nd'].append(valid_ep_out['metric']['acc_2nd'])
        valid_out['metric']['acc'].append(valid_ep_out['metric']['acc'])
        if valid_out['metric']['acc'][-1] > best_valid_acc:
            best_valid_acc = valid_out['metric']['acc'][-1]
            save_state_dict(model, save_folder, 'best-model.pth')
            logger.info("Saving Best...")

    return {
        'train': train_out,
        'valid': valid_out,
    }


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    model.train()

    losses = {
        'loss_mse': [],
        'loss_1st': [],
        'loss_2nd': [],
        'loss': [],
    }
    total_sample = 0
    total_density_map = 0
    total_mae = 0
    total_mse = 0
    total_correct_1st = 0
    total_correct_2nd = 0
    total_correct = 0
    for image, (density, d_mask), (label_1st, label_2nd, label) in data_loader:
        total_sample += len(label)
        total_density_map += torch.sum(torch.ones_like(d_mask)[d_mask]).numpy()
        image = image.to(device)
        density, d_mask = density.to(device), d_mask.to(device)
        label_1st, label_2nd, label = label_1st.to(device), label_2nd.to(device), label.to(device)

        density_map, out, out_1st, out_2nd = model(image, density, d_mask)
        loss_mse = torch.nn.MSELoss()(density_map[d_mask], density[d_mask])
        loss_1st = loss_fn(out_1st, label_1st)
        loss_2nd = loss_fn(out_2nd, label_2nd)
        loss = loss_fn(out, label)
        total_loss = loss_mse + loss_1st + loss_2nd + loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses['loss_mse'].append(loss_mse.detach().cpu().numpy())
        losses['loss_1st'].append(loss_1st.detach().cpu().numpy())
        losses['loss_2nd'].append(loss_2nd.detach().cpu().numpy())
        losses['loss'].append(loss.detach().cpu().numpy())
        total_mae += torch.sum(torch.abs(density_map[d_mask] - density[d_mask])).detach().cpu().numpy()
        total_mse += torch.sum((density_map[d_mask] - density[d_mask])**2).detach().cpu().numpy()
        total_correct_1st += (torch.argmax(out_1st, 1) == label_1st).int().sum().detach().cpu().numpy()
        total_correct_2nd += (torch.argmax(out_2nd, 1) == label_2nd).int().sum().detach().cpu().numpy()
        total_correct += (torch.argmax(out, 1) == label).int().sum().detach().cpu().numpy()
    scheduler.step()

    metric = {
        'mae': total_mae / total_density_map,
        'mse': total_mse / total_density_map,
        'acc_1st': total_correct_1st / total_sample,
        'acc_2nd': total_correct_2nd / total_sample,
        'acc': total_correct / total_sample,
    }
    return {
        'losses': losses,
        'metric': metric,
    }


@torch.no_grad()
def valid_epoch(model, data_loader, loss_fn, device):
    model.eval()

    losses = {
        'loss_mse': [],
        'loss_1st': [],
        'loss_2nd': [],
        'loss': [],
    }
    total_sample = 0
    total_density_map = 0
    total_mae = 0
    total_mse = 0
    total_correct_1st = 0
    total_correct_2nd = 0
    total_correct = 0
    for image, (density, d_mask), (label_1st, label_2nd, label) in data_loader:
        total_sample += len(label)
        total_density_map += torch.sum(torch.ones_like(d_mask)[d_mask]).numpy()
        image = image.to(device)
        density, d_mask = density.to(device), d_mask.to(device)
        label_1st, label_2nd, label = label_1st.to(device), label_2nd.to(device), label.to(device)

        density_map, out, out_1st, out_2nd = model(image)
        loss_mse = torch.nn.MSELoss()(density_map[d_mask], density[d_mask])
        loss_1st = loss_fn(out_1st, label_1st)
        loss_2nd = loss_fn(out_2nd, label_2nd)
        loss = loss_fn(out, label)

        losses['loss_mse'].append(loss_mse.cpu().numpy())
        losses['loss_1st'].append(loss_1st.cpu().numpy())
        losses['loss_2nd'].append(loss_2nd.cpu().numpy())
        losses['loss'].append(loss.cpu().numpy())
        total_mae += torch.sum(torch.abs(density_map[d_mask] - density[d_mask])).cpu().numpy()
        total_mse += torch.sum((density_map[d_mask] - density[d_mask]) ** 2).cpu().numpy()
        total_correct_1st += (torch.argmax(out_1st, 1) == label_1st).int().sum().cpu().numpy()
        total_correct_2nd += (torch.argmax(out_2nd, 1) == label_2nd).int().sum().cpu().numpy()
        total_correct += (torch.argmax(out, 1) == label).int().sum().cpu().numpy()

    metric = {
        'mae': total_mae / total_density_map,
        'mse': total_mse / total_density_map,
        'acc_1st': total_correct_1st / total_sample,
        'acc_2nd': total_correct_2nd / total_sample,
        'acc': total_correct / total_sample,
    }
    return {
        'losses': losses,
        'metric': metric,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--valid_size', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_fn', type=str, choices=('ce', 'focal'), default='ce')
    parser.add_argument('--optim', type=str, choices=('adam', 'sgd'), default='adam')
    parser.add_argument('--run_folder', type=str, default='./run/acc')

    arguments = parser.parse_args()
    main(arguments)
