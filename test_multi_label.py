#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from models import MultiLabelNet18
from utils import Logger, load_state_dict, accuracy, plus_or_minus_1_accuracy, \
    confusion_matrix, plot_confusion_matrix
from utils.data_trans import image_density_test_trans
from utils.dataset import ImageWithMultiLabel

SUMMARY_ITEMS = [
    'run_id',
    'train-acc',
    'test-acc',
    'train-±1acc',
    'test-±1acc',
    'train-acc_1st',
    'test-acc_1st',
    'train-acc_2nd',
    'test-acc_2nd',
]


def main(args):
    base_folder = args.run_folder
    run_ids = [exp_id for exp_id in os.listdir(base_folder) if 'EXP' in exp_id]
    device = torch.device(config.DEVICE)
    logger = Logger(base_folder)
    logger.info("Evaluation of Acne Severity Grading by Multi-level Category Labels: "
                f"run by {config.EXP_RUNNER} on {device}")

    # Data Preparation
    train_dataset = ImageWithMultiLabel(config.TRAIN_CSV_PATH,
                                        config.IMAGE_DIR,
                                        transform=image_density_test_trans)
    test_dataset = ImageWithMultiLabel(config.TEST_CSV_PATH,
                                       config.IMAGE_DIR,
                                       transform=image_density_test_trans)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    logger.info(f"ImageWithDensity {len(train_dataset)} train samples "
                f"and {len(test_dataset)} test samples")

    # Model Preparation
    model = MultiLabelNet18(3, config.NUM_1ST_LEVEL_CLASSES,
                            config.NUM_2ND_LEVEL_CLASSES,
                            config.NUM_CLASSES).to(device)
    logger.info("Using model MultiLabelNet18")

    # Evaluation Preparation
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
        train_result = evaluate(model, train_loader, device)
        test_result = evaluate(model, test_loader, device)
        logger.info(f"Evaluation Result: TRAIN acc {train_result['acc']:>5.3f} ±1acc {train_result['+-acc']:>5.3f} "
                    f"1st acc {train_result['acc_1st']:>5.3f} 2nd acc {train_result['acc_2nd']:>5.3f} | "
                    f"TEST acc {test_result['acc']:>5.3f} ±1acc {test_result['+-acc']:>5.3f} "
                    f"1st acc {test_result['acc_1st']:>5.3} 2nd acc {test_result['acc_2nd']:>5.3}")

        # Save Result
        summary['train-acc'].append(train_result["acc"])
        summary['train-±1acc'].append(train_result["+-acc"])
        summary['train-acc_1st'].append(train_result['acc_1st'])
        summary['train-acc_2nd'].append(train_result['acc_2nd'])
        summary['test-acc'].append(test_result["acc"])
        summary['test-±1acc'].append(test_result["+-acc"])
        summary['test-acc_1st'].append(test_result['acc_1st'])
        summary['test-acc_2nd'].append(test_result['acc_2nd'])

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


def plot_train_curve(curves, title=None, filename='loss_acc_curve.png'):
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(curves['train']['loss'], '.-', label='train loss', color='#FF7644')
    plt.plot(curves['train']['loss_1st'], '.-', label='train 1st loss', color='#FF7644', alpha=0.6)
    plt.plot(curves['train']['loss_2nd'], '.-', label='train 2nd loss', color='#FF7644', alpha=0.3)
    plt.plot(curves['valid']['loss'], '.-', label='valid loss', color='#F59E0B')
    plt.plot(curves['valid']['loss_1st'], '.-', label='valid 1st loss', color='#F59E0B', alpha=0.6)
    plt.plot(curves['valid']['loss_2nd'], '.-', label='valid 2nd loss', color='#F59E0B', alpha=0.3)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.ylabel('LOSS')

    plt.subplot(2, 1, 2)
    plt.plot(curves['train']['acc'], '.-', label='train acc', color='#D42BE6')
    plt.plot(curves['train']['acc_1st'], '.-', label='train 1st acc', color='#D42BE6', alpha=0.6)
    plt.plot(curves['train']['acc_2nd'], '.-', label='train 2nd acc', color='#D42BE6', alpha=0.3)
    plt.plot(curves['valid']['acc'], '.-', label='train acc', color='#8848FF')
    plt.plot(curves['valid']['acc_1st'], '.-', label='train 1st acc', color='#8848FF', alpha=0.6)
    plt.plot(curves['valid']['acc_2nd'], '.-', label='train 2nd acc', color='#8848FF', alpha=0.3)
    plt.legend(loc='lower right')
    plt.xlabel('EPOCHS')
    plt.ylabel('ACC')

    fig = plt.gcf()
    fig.savefig(filename)
    plt.show()
    plt.ioff()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    preds = []
    targets = []
    preds_1st = []
    targets_1st = []
    preds_2nd = []
    targets_2nd = []
    for images, (labels_1st, labels_2nd, labels) in data_loader:
        images = images.to(device)
        labels_1st, labels_2nd, labels = labels_1st.to(device), labels_2nd.to(device), labels.to(device)

        out_1st, out_2nd, out = model(images)

        preds.append(torch.argmax(out, dim=1))
        targets.append(labels)
        preds_1st.append(torch.argmax(out_1st, dim=1))
        targets_1st.append(labels_1st)
        preds_2nd.append(torch.argmax(out_2nd, dim=1))
        targets_2nd.append(labels_2nd)
    preds = torch.cat(preds, dim=-1).cpu().numpy()
    targets = torch.cat(targets, dim=-1).cpu().numpy()
    preds_1st = torch.cat(preds_1st, dim=-1)
    targets_1st = torch.cat(targets_1st, dim=-1)
    preds_2nd = torch.cat(preds_2nd, dim=-1)
    targets_2nd = torch.cat(targets_2nd, dim=-1)

    return {
        'acc': accuracy(preds, targets),
        '+-acc': plus_or_minus_1_accuracy(preds, targets),
        'acc-1st': accuracy(preds_1st, targets_1st),
        'acc-2nd': accuracy(preds_2nd, targets_2nd),
        'c_matrix': confusion_matrix(preds, targets),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--run_folder', type=str, default='./run/multi_label')

    arguments = parser.parse_args()
    main(arguments)
