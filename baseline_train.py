#!/usr/bin/env python
# coding: utf-8
# 基础模型训练
# 指定gpu请设置环境变量 CUDA_VISIBLE_DEVICES
import os
import datetime
import argparse

import torch

from models.baseline import ResNet50Baseline
from models.loss_fn import get_loss_fn
from utils.dataset import AcneImageDataset
from utils.data_trans import BASIC_TRAIN_TRANS
from utils import save_state_dict, save_result, Logger
from share import train_valid_split, get_optimizer, train_and_valid


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
    logger.info(f"Baseline Ex({run_id}) run by Chen: train on {device}")

    # Data Preparation
    dataset = AcneImageDataset('./data/HX_Acne_Image_GroundTruth_Train.csv',
                               './data/images',
                               transform=BASIC_TRAIN_TRANS)
    train_loader, valid_loader = train_valid_split(dataset, args.val_size, args.batch_size)

    # Model Preparation
    model = ResNet50Baseline().to(device)

    # Training Preparation
    criterion = get_loss_fn(args.loss, reduction=None)
    optimizer = get_optimizer(args.opt, model.parameters(), args.lr)

    # Train
    logger.info("Training parameters:")
    logger.info(f"batch_size      {args.batch_size}")
    logger.info(f"val_size        {args.val_size}")
    logger.info(f"epochs          {args.epochs}")
    logger.info(f"early_threshold {args.early_threshold}")
    logger.info(f"model           ResNet50Baseline")
    logger.info(f"criterion       {args.loss}")
    logger.info(f"learning_rate   {args.lr}")
    logger.info(f"optimizer       {args.opt}")

    result = train_and_valid(model, criterion, optimizer,
                             train_loader, valid_loader,
                             args.epochs, args.early_threshold,
                             model_folder, "first-best-model.pth",
                             logger, device)

    save_state_dict(model, model_folder, "final-model.pth")
    save_result({
        "train_param": f"{args.loss}-base",
        "train_loss": result['train_loss'],
        "train_acc": result['train_acc'],
        "valid_loss": result['valid_loss'],
        "valid_acc": result['valid_acc'],
    }, run_folder, "result.pkl")
    logger.info(f"Baseline Ex({run_id}) is over!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--run_folder', type=str, default='./run/baseline')
    parser.add_argument('--early_threshold', type=int, default=40)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--loss', type=str, default='ce', choices=('ce', 'focal'))
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))

    arguments = parser.parse_args()
    main(arguments)
