from matplotlib import pyplot as plt


def plot_acc(train_accuracy, val_accuracy):
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validataion accuracy')
    plt.legend()
    plt.grid()
    plt.show()