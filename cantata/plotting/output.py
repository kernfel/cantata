import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_voltage_traces(mem, spk=None, dim=(3,5), spike_height=5, ax=None, fig=None):
    if ax is None:
        if fig is None:
            _,ax = plt.subplots(*dim, sharey=True)
        else:
            ax = fig.subplots(*dim, sharey=True)
    ax = ax.flatten()
    if spk is not None:
        dat = (mem+spike_height*spk).detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        ax[i].plot(dat[i])
        ax[i].axis("off")
    return ax
