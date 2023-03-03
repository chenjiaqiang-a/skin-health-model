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
from utils.dataset import AcneImageDataset, MappingDataset
from utils.data_trans import BASIC_TRAIN_TRANS
from utils import save_state_dict, save_result, Logger, load_state_dict
from share import get_optimizer, train_valid_split, train_and_valid


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
    logger.info(f"Layered Ex({run_id}) run by Chen: train on {device}")

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

    dataset = MappingDataset('./data/HX_Acne_Image_GroundTruth_Train.csv',
                             './data/images',
                             transform=BASIC_TRAIN_TRANS)
    train_loader, valid_loader = train_valid_split(dataset, args.val_size, args.batch_size)

    logger.info("First Training...")
    logger.info(f"Class Mapping: {dataset.mapping}")
    first_result = train_and_valid(model, criterion, optimizer,
                                   train_loader, valid_loader,
                                   args.epochs, args.early_threshold,
                                   model_folder, "first-best-model.pth",
                                   logger, device)

    load_state_dict(model, os.path.join(model_folder, 'first-best-model.pth'), device)

    dataset = AcneImageDataset('./data/HX_Acne_Image_GroundTruth_Train.csv',
                               './data/images',
                               transform=BASIC_TRAIN_TRANS)
    train_loader, valid_loader = train_valid_split(dataset, args.val_size, args.batch_size)

    logger.info("Second Training...")
    second_result = train_and_valid(model, criterion, optimizer,
                                    train_loader, valid_loader,
                                    args.epochs, args.early_threshold,
                                    model_folder, "second-best-model.pth",
                                    logger, device)

    save_state_dict(model, model_folder, 'second-final-model.pth')
    save_result({
        "train_param": f"{args.loss}-base",
        "first-train_loss": first_result['train_loss'],
        "first-train_acc": first_result['train_acc'],
        "first-valid_loss": first_result['valid_loss'],
        "first-valid_acc": first_result['valid_acc'],
        "second-train_loss": second_result['train_loss'],
        "second-train_acc": second_result['train_acc'],
        "second-valid_loss": second_result['valid_loss'],
        "second-valid_acc": second_result['valid_acc'],
    }, run_folder, "result.pkl")
    logger.info(f"Layered Ex({run_id}) is over!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--run_folder', type=str, default='./run/layered')
    parser.add_argument('--early_threshold', type=int, default=20)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--loss', type=str, default='ce', choices=('ce', 'focal'))
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))

    arguments = parser.parse_args()
    main(arguments)
