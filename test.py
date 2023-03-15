import numpy as np
import pandas as pd

from utils import plot_confusion_matrix
from utils.dataset import ACNE_CATEGORIES

df = pd.read_csv('best-train-c-matrix.csv')
cm = np.array(df)
print(cm)
plot_confusion_matrix(cm, ACNE_CATEGORIES, 'c matrix', 'c-matrix.png')
