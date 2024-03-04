import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.show()
