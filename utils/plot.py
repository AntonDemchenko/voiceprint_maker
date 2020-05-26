import sys

import matplotlib.pyplot as plt
import pandas as pd


def draw_loss_acc_plots(log_path):
    plot_df = pd.read_csv(log_path)
    fig, axs = plt.subplots(2, sharex=True)
    epochs = list(range(1, len(plot_df) + 1))
    label = dict(accuracy='Accuracy', loss='Loss')
    for value, ax in zip(['accuracy', 'loss'], axs):
        ax.plot(epochs, plot_df[value], label='train ' + label[value])
        ax.plot(epochs, plot_df['val_{}'.format(value)], label='validation ' + label[value])
        ax.set(xlabel='Epochs', ylabel=label[value])
        ax.grid(True)
        ax.legend()
    plt.show()


if __name__ == '__main__':
    log_path = sys.argv[1]
    draw_loss_acc_plots(log_path)
