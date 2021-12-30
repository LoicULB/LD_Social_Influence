import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_curve(scores, figure_file):
    nb_steps = len(scores[0])
    x_plot = [i + 1 for i in range(nb_steps)]
    y_plot = np.array(scores)
    y_mean = np.mean(y_plot, axis=0)
    y_stv = np.std(y_plot, axis=0)
    plt.plot(x, y_mean)
    plt.title('Collective reward (mean)')
    plt.savefig(figure_file)
