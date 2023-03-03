#!/usr/bin/env python
# coding: utf-8
# 基础模型测试
# 指定gpu请设置环境变量 CUDA_VISIBLE_DEVICES
import os
import pickle
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.baseline import ResNet50Baseline
from utils.dataset import AcneImageDataset
from utils.data_trans import BASIC_TEST_TRANS
from utils import load_state_dict, Logger, Evaluation, plot_confusion_matrix, plot_train_curve

SUMMARY_ITEMS = [
    'run_id',
    'train_param',
    'final-train-acc',
    'final-test-acc',
    'best-train-acc',
    'best-test-acc',
]
DETAIL_ITEMS = [
    'final-train-precision',
    'final-train-recall',
    'final-train-f1',
    'final-test-precision',
    'final-test-recall',
    'final-test-f1',
    'best-train-precision',
    'best-train-recall',
    'best-train-f1',
    'best-test-precision',
    'best-test-recall',
    'best-test-f1',
]


def main(args):
    base_folder = args.run_folder
    run_ids = os.listdir(base_folder)
    logger = Logger(base_folder, "test")
    device = torch.device('cuda:0')
    logger.info(f"Evaluation run by Chen: Baseline test on {device}")

    # Data Preparation
    train_dataset = AcneImageDataset('./data/HX_Acne_Image_GroundTruth_Train.csv',
                                     './data/images',
                                     transform=BASIC_TEST_TRANS)
    test_dataset = AcneImageDataset('./data/HX_Acne_Image_GroundTruth_Test.csv',
                                    './data/images',
                                    transform=BASIC_TEST_TRANS)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    logger.info(f"Test on HX Skin DAtaset(Acne) with {len(train_dataset)} train samples, "
                f"{len(test_dataset)} test samples")

    # Model Preparation
    model = ResNet50Baseline()

    # Evaluation
    evaluation = Evaluation(device=device)
    summary = {
        item: [] for item in SUMMARY_ITEMS
    }
    for run_id in run_ids:
        detail = {
            item: [] for item in DETAIL_ITEMS
        }
        run_folder = os.path.join(base_folder, run_id)
        model_folder = os.path.join(run_folder, 'models')
        image_folder = os.path.join(run_folder, 'images')
        with open(os.path.join(run_folder, 'result.pkl'), 'rb') as fp:
            train_result = pickle.load(fp)
        summary['run_id'].append(run_id)
        summary['train_param'].append(train_result['train_param'])
        # plot train curve
        plot_train_curve(train_result, 'training curves', os.path.join(image_folder, 'acc_loss_curve.png'))

        # Test Final
        load_state_dict(model, os.path.join(model_folder, 'final-model.pth'))
        train_out = evaluation.evaluate(['acc', 'precision', 'recall', 'f1_score', 'c_matrix'], model, train_loader)
        test_out = evaluation.evaluate(['acc', 'precision', 'recall', 'f1_score', 'c_matrix'], model, test_loader)
        plot_confusion_matrix(train_out["c_matrix"],
                              train_dataset.categories,
                              'c matrix for train data',
                              os.path.join(image_folder, 'final-train-c-matrix.png'))
        plot_confusion_matrix(test_out["c_matrix"],
                              test_dataset.categories,
                              'c matrix for test data',
                              os.path.join(image_folder, 'final-test-c-matrix.png'))
        summary['final-train-acc'].append(train_out["acc"])
        summary['final-test-acc'].append(test_out["acc"])
        detail['final-train-precision'] = train_out['precision']
        detail['final-train-recall'] = train_out['recall']
        detail['final-train-f1'] = train_out['f1_score']
        detail['final-test-precision'] = test_out['precision']
        detail['final-test-recall'] = test_out['recall']
        detail['final-test-f1'] = test_out['f1_score']

        # Test Best
        load_state_dict(model, os.path.join(model_folder, 'best-model.pth'))
        train_out = evaluation.evaluate(['acc', 'precision', 'recall', 'f1_score', 'c_matrix'], model, train_loader)
        test_out = evaluation.evaluate(['acc', 'precision', 'recall', 'f1_score', 'c_matrix'], model, test_loader)
        plot_confusion_matrix(train_out["c_matrix"],
                              train_dataset.categories,
                              'c matrix for train data',
                              os.path.join(image_folder, 'best-train-c-matrix.png'))
        plot_confusion_matrix(test_out["c_matrix"],
                              test_dataset.categories,
                              'c matrix for test data',
                              os.path.join(image_folder, 'best-test-c-matrix.png'))
        summary['best-train-acc'].append(train_out["acc"])
        summary['best-test-acc'].append(test_out["acc"])
        detail['best-train-precision'] = train_out['precision']
        detail['best-train-recall'] = train_out['recall']
        detail['best-train-f1'] = train_out['f1_score']
        detail['best-test-precision'] = test_out['precision']
        detail['best-test-recall'] = test_out['recall']
        detail['best-test-f1'] = test_out['f1_score']
        df = pd.DataFrame.from_dict(detail, orient='index', columns=train_dataset.categories)
        df.to_csv(os.path.join(run_folder, 'evaluation.csv'))
    df = pd.DataFrame.from_dict(summary)
    df.to_csv(os.path.join(base_folder, 'evaluation.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--run_folder', type=str, default='./run/baseline')

    arguments = parser.parse_args()
    main(arguments)
