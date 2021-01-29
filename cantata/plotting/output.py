import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cantata
from cantata import cfg

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

def raster(spikes, ax = None, rate = None):
    total = 0
    ticks, lticks = [0], []
    labels = []
    N = cantata.init.get_N()
    if ax == None:
        fig, ax = plt.subplots(figsize=(20,15))
    plt.imshow(spikes.T, cmap=plt.cm.gray_r, origin='lower')
    for name, pop in cfg.model.populations.items():
        lticks.append(total + pop.n/2)
        labels.append(name)
        total += pop.n
        ticks.append(total)
    if rate is not None:
        if type(rate) == dict:
            plotln = lambda k: plt.plot(k, **rate)
        elif type(rate) == str:
            plotln = lambda k: plt.plot(k, rate)
        else:
            plotln = lambda k: plt.plot(k)
        for i,j in zip(ticks[:-1], ticks[1:]):
            rates = np.sum(spikes[:,i:j], axis=1)
            plotln(i + rates * 0.9*(j-i) / rates.max())
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_yticks(lticks, minor=True)
    ax.set_yticklabels(labels, minor=True)
    ax.yaxis.set_tick_params(which='minor', length=0)
    ax.set_ylim(0, ticks[-1])
    ax.set_xlim(0, spikes.shape[0])
    for i in range(1, len(ticks)-1, 2):
        ax.axhspan(ticks[i], ticks[i+1], fc='gray', alpha=0.2)
    sns.despine()
    return ax
