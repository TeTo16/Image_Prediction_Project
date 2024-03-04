from matplotlib import pyplot as plt


def plot_losses(train_loss, val_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.legend()
    plt.grid()
    plt.show()