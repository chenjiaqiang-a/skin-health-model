#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from models import ResNet18
from utils.dataset import AcneDataset
from utils.data_trans import BASIC_TEST_TRANS
from utils import load_state_dict, Evaluation, plot_confusion_matrix, plot_train_curve, Logger

SUMMARY_ITEMS = [
    'run_id',
    'train-acc',
    'test-acc',
    'train-±1acc',
    'test-±1acc',
]


def main(args):
    base_folder = args.run_folder
    run_ids = [exp_id for exp_id in os.listdir(base_folder) if 'EXP' in exp_id]
    device = torch.device(config.DEVICE)
    logger = Logger(base_folder)
    logger.info(f"Evaluation of Acne Severity Grading Baseline: run by {config.EXP_RUNNER} on {device}")

    # Data Preparation
    train_dataset = AcneDataset(config.TRAIN_CSV_PATH,
                                config.IMAGE_DIR,
                                transform=BASIC_TEST_TRANS)
    test_dataset = AcneDataset(config.TEST_CSV_PATH,
                               config.IMAGE_DIR,
                               transform=BASIC_TEST_TRANS)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    logger.info(f"AcneDataset {len(train_dataset)} train samples "
                f"and {len(test_dataset)} test samples")

    # Model Preparation
    model = ResNet18(config.NUM_CLASSES).to(device)
    logger.info(f"Using model {model}")

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

        # Evaluate
        load_state_dict(model, os.path.join(model_folder, 'best-model.pth'))
        train_result = evaluation.evaluate(['acc', '+-acc', 'c_matrix'], model, train_loader)
        test_result = evaluation.evaluate(['acc', '+-acc', 'c_matrix'], model, test_loader)
        logger.info(f"Evaluation Result: TRAIN acc {train_result['acc']:>5.3f} ±1acc {train_result['+-acc']:>5.3f} | "
                    f"TEST acc {test_result['acc']:>5.3f} ±1acc {test_result['+-acc']:>5.3f}")

        # Save Result
        summary['train-acc'].append(train_result["acc"])
        summary['train-±1acc'].append(train_result["+-acc"])
        summary['test-acc'].append(test_result["acc"])
        summary['test-±1acc'].append(test_result["+-acc"])

        plot_confusion_matrix(train_result["c_matrix"],
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--run_folder', type=str, default='./run/baseline')

    arguments = parser.parse_args()
    main(arguments)
