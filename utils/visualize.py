from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors

__all__ = ['plot_confusion_matrix', 'plot_roc_curves', 'plot_train_curve']


# evaluation
def plot_confusion_matrix(cm, classes, title=None, filename="cm.png", cmap="Blues"):
    plt.rc('font', size='8')  # 设置字体大小

    # 按行进行归一化
    cm_img = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10), facecolor='w')
    fig, ax = plt.subplots()
    im = ax.imshow(cm_img, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='Actual',
           xlabel='Predicted')
    ax.set_xticklabels(labels=classes, rotation=45)
    ax.set_yticklabels(labels=classes, rotation=45)

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 标注百分比信息
    fmt = 'd'
    thresh = cm_img.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_img[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.clf()


def plot_roc_curves(fpr, tpr, roc_auc, categories=None, title=None, filename="roc.png"):
    # 绘制全部的ROC曲线
    n_classes = len(fpr) - 2

    plt.figure(facecolor='w')
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(list(m_colors.TABLEAU_COLORS.keys()))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=m_colors.TABLEAU_COLORS[color], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(categories[i] if categories else i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()


def plot_train_curve(curves, title=None, filename="loss_acc_curve.png"):
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(curves['train']['loss'], '.-', label='train loss', color='#FF0000')
    plt.plot(curves['valid']['loss'], '.-', label='valid loss', color='#D2691E')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(curves['train']['acc'], '.-', label='train acc', color='#0000FF')
    plt.plot(curves['valid']['acc'], '.-', label='valid acc', color='#FF0000')
    plt.legend(loc='upper right')
    plt.xlabel('EPOCHS')
    plt.ylabel('Acc')

    fig = plt.gcf()
    fig.savefig(filename)
    plt.show()
    plt.ioff()
