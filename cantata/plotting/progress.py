import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

def get_grad_plotter(alpha = 0.1, top = 0.02):
    def plot_grads(named_parameters, ax):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad:
                grad = p.grad.abs()
                ave_grads.append(grad.mean())
                max_grads.append(grad.max())
                layers.append(n)
        ax.bar(np.arange(len(max_grads))+.4, max_grads, alpha=alpha, width=.4, color="c")
        ax.bar(np.arange(len(max_grads)), ave_grads, alpha=alpha, width=.4, color="b")
        ax.xaxis.set_ticks(range(0,len(ave_grads), 1))
        ax.xaxis.set_ticklabels(layers, rotation="vertical")
        ax.set_ylim(top=top)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient")
        ax.grid(True)
        ax.legend([Line2D([0], [0], color="c", lw=4),
                   Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])
        sns.despine(offset=4)
    return plot_grads
