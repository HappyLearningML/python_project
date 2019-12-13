#-*-coding:utf-8-*-
import matplotlib.pyplot as plt

def plot_trainloss(epoch, avg_per_epoch, title, xlabel, ylabel, savename,picsize=(11,8)):

    fig, ax = plt.subplots(figsize=picsize)

    ax.plot(range(epoch+1), avg_per_epoch)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(savename)

    return fig