import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cantata
from cantata import cfg
import torch

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

def raster(spikes, ax = None, rates = None, **kwargs):
    if spikes.shape[0] > 1:
        for b in range(spikes.shape[0]):
            raster(spikes[b:b+1], ax, rates if rates is None else rates[b:b+1])
    ticks, lticks, labels = get_ticks()
    if ax == None:
        fig, ax = plt.subplots(figsize=(20,10))
    plt.imshow(spikes[0].cpu().T, cmap=plt.cm.gray_r, origin='lower',
        aspect='auto', interpolation='none')
    if rates is not None:
        for i,(lo,hi) in enumerate(zip(ticks[:-1], ticks[1:])):
            r = rates[0,i].cpu().numpy()
            if r.max() > 0:
                plt.plot(lo + r * 0.9*(hi-lo) / r.max(), **kwargs)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_yticks(lticks, minor=True)
    ax.set_yticklabels(labels, minor=True)
    ax.yaxis.set_tick_params(which='minor', length=0)
    for i in range(1, len(ticks)-1, 2):
        ax.axhspan(ticks[i], ticks[i+1], fc='gray', alpha=0.2)
    sns.despine()
    return ax

def get_ticks():
    total = 0
    ticks, lticks = [0], []
    labels = []
    for name, pop in cfg.model.populations.items():
        lticks.append(total + pop.n/2)
        labels.append(name)
        total += pop.n
        ticks.append(total)
    return ticks, lticks, labels

def get_rates(spikes, kwidth = 0):
    ticks,_,_ = get_ticks()
    rates = torch.empty(cfg.batch_size, len(ticks)-1, spikes.shape[1], **cfg.tspec)
    for i,(lo,hi) in enumerate(zip(ticks[:-1], ticks[1:])):
        rates[:,i,:] = torch.sum(spikes[:,:,lo:hi], axis=2) / (hi-lo) / cfg.time_step
    if kwidth > 0:
        if not kwidth%2:
            kwidth += 1
        kernel = torch.ones(len(ticks)-1, 1, kwidth, **cfg.tspec) / kwidth
        conv_rates = torch.nn.functional.conv1d(rates, kernel,
            groups = len(ticks)-1, padding = int((kwidth-1)/2))
        return conv_rates
    else:
        return rates
