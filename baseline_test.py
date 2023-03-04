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
from utils import load_state_dict, Evaluation, plot_confusion_matrix, plot_train_curve

SUMMARY_ITEMS = [
    'run_id',
    'train_param',
    'final-train-acc',
    'final-test-acc',
    'best-train-acc',
    'best-test-acc',
]


def main(args):
    base_folder = args.run_folder
    run_ids = os.listdir(base_folder)
    device = torch.device('cuda:0')

    # Data Preparation
    train_dataset = AcneImageDataset('./data/HX_Acne_Image_GroundTruth_Train.csv',
                                     './data/images',
                                     transform=BASIC_TEST_TRANS)
    test_dataset = AcneImageDataset('./data/HX_Acne_Image_GroundTruth_Test.csv',
                                    './data/images',
                                    transform=BASIC_TEST_TRANS)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Model Preparation
    model = ResNet50Baseline()

    # Evaluation
    evaluation = Evaluation(device=device)
    summary = {
        item: [] for item in SUMMARY_ITEMS
    }
    for run_id in run_ids:
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
        train_out = evaluation.evaluate(['acc', 'c_matrix'], model, train_loader)
        test_out = evaluation.evaluate(['acc', 'c_matrix'], model, test_loader)
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
        df = pd.DataFrame(train_out['c_matrix'],
                          index=train_dataset.categories,
                          columns=train_dataset.categories)
        df.to_csv(os.path.join(run_folder, 'best-train-c-matrix.csv'), index=False)
        df = pd.DataFrame(test_out['c_matrix'],
                          index=train_dataset.categories,
                          columns=train_dataset.categories)
        df.to_csv(os.path.join(run_folder, 'best-test-c-matrix.csv'), index=False)

        # Test Best
        load_state_dict(model, os.path.join(model_folder, 'best-model.pth'))
        train_out = evaluation.evaluate(['acc', 'c_matrix'], model, train_loader)
        test_out = evaluation.evaluate(['acc', 'c_matrix'], model, test_loader)
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
        df = pd.DataFrame(train_out['c_matrix'],
                          index=train_dataset.categories,
                          columns=train_dataset.categories)
        df.to_csv(os.path.join(run_folder, 'best-train-c-matrix.csv'), index=False)
        df = pd.DataFrame(test_out['c_matrix'],
                          index=train_dataset.categories,
                          columns=train_dataset.categories)
        df.to_csv(os.path.join(run_folder, 'best-test-c-matrix.csv'), index=False)
    df = pd.DataFrame.from_dict(summary)
    df.to_csv(os.path.join(base_folder, 'evaluation.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--run_folder', type=str, default='./run/baseline')

    arguments = parser.parse_args()
    main(arguments)
