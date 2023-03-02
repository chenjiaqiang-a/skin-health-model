#!/usr/bin/env python
# coding: utf-8
# 基础模型训练
# 指定gpu请设置环境变量 CUDA_VISIBLE_DEVICES
import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.baseline import ResNet50Baseline
from utils.dataset import AcneImageDataset
from utils.data_trans import BASIC_TEST_TRANS
from utils import load_state_dict


def main(args):
    base_folder = "./run/baseline"
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
    model = ResNet50Baseline().to(device)

    summary = {
        'run_id': [],
        'final-train-acc': [],
        'final-train-precision': [],
        'final-train-recall': [],
        'final-train-f1': [],
        'final-test-acc': [],
        'final-test-precision': [],
        'final-test-recall': [],
        'final-test-f1': [],
        'best-train-acc': [],
        'best-train-precision': [],
        'best-train-recall': [],
        'best-train-f1': [],
        'best-test-acc': [],
        'best-test-precision': [],
        'best-test-recall': [],
        'best-test-f1': [],
    }
    for run_id in run_ids:
        run_folder = os.path.join(base_folder, run_id)
        model_folder = os.path.join(run_folder, 'models')
        image_folder = os.path.join(run_folder, 'images')
        with open(os.path.join(run_folder, 'result.pkl'), 'rb') as fp:
            train_result = pickle.load(fp)
        # plot train curve
        # plot_curve()

        # Test Final
        load_state_dict(model, os.path.join(model_folder, 'final-model.pth'))
        # train_result = evaluate(model, train_loader, device)
        # test_result = evaluate(model, test_loader, device)
        # plot_c_matrix(train_result["c_matrix"], 'train', os.path.join(image_folder, 'final-train-c-matrix.png'))
        # plot_c_matrix(test_result["c_matrix"], 'test', os.path.join(image_folder, 'final-test-c-matrix.png'))
        # summary['final-train-acc'].append(train_result["acc"])
        # summary['final-train-precision'].append(train_result['precision'])
        # summary['final-train-recall'].append(train_result['recall'])
        # summary['final-train-f1'].append(train_result['f1'])
        # summary['final-test-acc'].append(test_result["acc"])
        # summary['final-test-precision'].append(test_result['precision'])
        # summary['final-test-recall'].append(test_result['recall'])
        # summary['final-test-f1'].append(test_result['f1'])

        # Test Best
        load_state_dict(model, os.path.join(model_folder, 'best-model.pth'))
        # train_result = evaluate(model, train_loader, device)
        # test_result = evaluate(model, test_loader, device)
        # plot_c_matrix(train_result["c_matrix"], 'train', os.path.join(image_folder, 'final-train-c-matrix.png'))
        # plot_c_matrix(test_result["c_matrix"], 'test', os.path.join(image_folder, 'final-test-c-matrix.png'))
        # summary['final-train-acc'].append(train_result["acc"])
        # summary['final-train-precision'].append(train_result['precision'])
        # summary['final-train-recall'].append(train_result['recall'])
        # summary['final-train-f1'].append(train_result['f1'])
        # summary['final-test-acc'].append(test_result["acc"])
        # summary['final-test-precision'].append(test_result['precision'])
        # summary['final-test-recall'].append(test_result['recall'])
        # summary['final-test-f1'].append(test_result['f1'])


def evaluate(model, data_iter, device):
    pass


if __name__ == '__main__':
    main()
