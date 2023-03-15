#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config
from models import DensityNet18
from utils import Logger, Evaluation, plot_train_curve, load_state_dict, accuracy, plus_or_minus_1_accuracy, \
    confusion_matrix, plot_confusion_matrix
from utils.data_trans import image_density_test_trans
from utils.dataset import ImageWithDensity

SUMMARY_ITEMS = [
    'run_id',
    'train-acc',
    'test-acc',
    'train-±1acc',
    'test-±1acc',
    'train-mae',
    'test-mae',
    'train-mse',
    'test-mse',
]


def main(args):
    base_folder = args.run_folder
    run_ids = [exp_id for exp_id in os.listdir(base_folder) if 'EXP' in exp_id]
    device = torch.device(config.DEVICE)
    logger = Logger(base_folder)
    logger.info("Evaluation of Acne Severity Grading by Density Map: "
                f"run by {config.EXP_RUNNER} on {device}")

    # Data Preparation
    train_dataset = ImageWithDensity(config.TRAIN_CSV_PATH,
                                     config.IMAGE_DIR,
                                     config.DENSITY_MAP_DIR,
                                     transform=image_density_test_trans)
    test_dataset = ImageWithDensity(config.TEST_CSV_PATH,
                                    config.IMAGE_DIR,
                                    config.DENSITY_MAP_DIR,
                                    transform=image_density_test_trans)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    logger.info(f"ImageWithDensity {len(train_dataset)} train samples "
                f"and {len(test_dataset)} test samples")

    # Model Preparation
    model = DensityNet18(config.NUM_CLASSES).to(device)
    logger.info("Using model DensityNet18")

    # Evaluation Preparation
    evaluation = Evaluation(device=device)
    summary = {
        item: [] for item in SUMMARY_ITEMS
    }

    for run_id in run_ids:
        logger.info(f"Evaluation of {run_id}")
        summary['run_id'].append(run_id)
        run_folder = os.path.join(base_folder, run_id)
        model_folder = os.path.join(run_folder, 'models')
        image_folder = os.path.join(run_folder, 'images')

        # Plot Training Curves
        with open(os.path.join(run_folder, 'result.pkl'), 'rb') as fp:
            train_output = pickle.load(fp)
        plot_train_curve(train_output, 'training curves', os.path.join(image_folder, 'loss_acc_curve.png'))
        plot_mae_mse_curve(train_output, 'training mse and mae', os.path.join(image_folder, 'mse_mae_curve.png'))

        # Evaluate
        load_state_dict(model, os.path.join(model_folder, 'best-model.pth'))
        train_result = evaluate(model, train_loader, device)
        test_result = evaluate(model, test_loader, device)
        logger.info(f"Evaluation Result: TRAIN acc {train_result['acc']:>5.3f} ±1acc {train_result['+-acc']:>5.3f} "
                    f"mae {train_result['mae']:>7.3f} mse {train_result['mse']:>6.3f} | "
                    f"TEST acc {test_result['acc']:>5.3f} ±1acc {test_result['+-acc']:>5.3f} "
                    f"mae {test_result['mae']:>7.3} mse {test_result['mse']:>6.3}")

        # Save Result
        summary['train-acc'].append(train_result["acc"])
        summary['train-±1acc'].append(train_result["+-acc"])
        summary['train-mae'].append(train_result['mae'])
        summary['train-mse'].append(train_result['mse'])
        summary['test-acc'].append(test_result["acc"])
        summary['test-±1acc'].append(test_result["+-acc"])
        summary['test-mae'].append(test_result['mae'])
        summary['test-mse'].append(test_result['mse'])

        plot_confusion_matrix(train_result['c_matrix'],
                              train_dataset.categories,
                              'c matrix for train data',
                              os.path.join(image_folder, 'train-c-matrix.png'))
        plot_confusion_matrix(test_result["c_matrix"],
                              test_dataset.categories,
                              'c matrix for test data',
                              os.path.join(image_folder, 'test-c-matrix.png'))
        df = pd.DataFrame(train_result['c_matrix'],
                          index=train_dataset.categories,
                          columns=train_dataset.categories)
        df.to_csv(os.path.join(run_folder, 'train-c-matrix.csv'))
        df = pd.DataFrame(test_result['c_matrix'],
                          index=train_dataset.categories,
                          columns=train_dataset.categories)
        df.to_csv(os.path.join(run_folder, 'test-c-matrix.csv'))

    df = pd.DataFrame.from_dict(summary)
    col_mean = df[SUMMARY_ITEMS[1:]].mean()
    col_mean[SUMMARY_ITEMS[0]] = 'mean'
    df = df.append(col_mean, ignore_index=True)
    df.to_csv(os.path.join(base_folder, 'evaluation.csv'), index=False)


def plot_mae_mse_curve(curves, title=None, filename='mae_mse_curve.png'):
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(curves['train']['loss_mse'], '.-', label='train mse loss', color='FF7644')
    plt.plot(curves['train']['mse'], '.-', label='train mse', color='FF7644', alpha=0.3)
    plt.plot(curves['valid']['loss_mse'], '.-', label='valid mse loss', color='F59E0B')
    plt.plot(curves['valid']['mse'], '.-', label='valid mse', color='F59E0B', alpha=0.3)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.ylabel('MSE')

    plt.subplot(2, 1, 2)
    plt.plot(curves['train']['mae'], '.-', label='train mae', color='D42BE6')
    plt.plot(curves['valid']['mae'], '.-', label='train mae', color='8848FF')
    plt.legend(loc='upper right')
    plt.xlabel('EPOCHS')
    plt.ylabel('MAE')

    fig = plt.gcf()
    fig.savefig(filename)
    plt.show()
    plt.ioff()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    preds = []
    targets = []
    total_density_sample = 0
    total_mae = 0
    total_mse = 0
    for images, (densities, d_masks), labels in data_loader:
        total_density_sample += torch.sum(torch.ones_like(d_masks)[d_masks]).numpy()
        images = images.to(device)
        densities, d_masks = densities.to(device), d_masks.to(device)
        labels = labels.to(device)

        density_out, out = model(images, densities, d_masks)

        total_mae += torch.sum(torch.abs(density_out[d_masks] - densities[d_masks])).detach().cpu().numpy()
        total_mse += torch.sum((density_out[d_masks] - densities[d_masks]) ** 2).detach().cpu().numpy()

        preds.append(torch.argmax(out, dim=1))
        targets.append(labels)
    preds = torch.cat(preds, dim=-1).cpu().numpy()
    targets = torch.cat(targets, dim=-1).cpu().numpy()

    return {
        'acc': accuracy(preds, targets),
        '+-acc': plus_or_minus_1_accuracy(preds, targets),
        'c_matrix': confusion_matrix(preds, targets),
        'mae': total_mae / total_density_sample,
        'mse': total_mse / total_density_sample,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--run_folder', type=str, default='./run/density')

    arguments = parser.parse_args()
    main(arguments)
