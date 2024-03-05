import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(labels, predictions):
    age_range = [f"{i}-{i + 4}" for i in range(0, 100, 5)]
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=age_range)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp.plot(ax=ax, cmap=plt.cm.OrRd)
    plt.show()
