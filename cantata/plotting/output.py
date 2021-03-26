import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cantata
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

def raster(spikes, conductor, ax = None, rates = None, **kwargs):
    '''
    Assumes spikes as (t,b,N) with N covering the entire model.
    Assumes rates as None, or (t,b,npop)
    '''
    if spikes.shape[1] > 1:
        for b in range(spikes.shape[1]):
            raster(spikes[:,b:b+1], conductor, ax,
                   rates if rates is None else rates[:,b:b+1])
    ticks, lticks, labels = get_ticks(conductor)
    if ax == None:
        fig, ax = plt.subplots(figsize=(20,10))
    plt.imshow(spikes[:,0].cpu().T, cmap=plt.cm.gray_r, origin='lower',
        aspect='auto', interpolation='none')
    if rates is not None:
        for i,(lo,hi) in enumerate(zip(ticks[:-1], ticks[1:])):
            r = rates[:,0,i].cpu().numpy()
            if r.max() > 0:
                plt.plot(lo + r * 0.9*(hi-lo) / r.max(), **kwargs)
            else:
                plt.plot([])
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_yticks(lticks, minor=True)
    ax.set_yticklabels(labels, minor=True)
    ax.yaxis.set_tick_params(which='minor', length=0)
    for i in range(1, len(ticks)-1, 2):
        ax.axhspan(ticks[i], ticks[i+1], fc='gray', alpha=0.1)
    sns.despine()
    return ax

def get_ticks(conductor):
    ticks, lticks = [0], []
    labels = []
    def add_area(area, ticks, lticks, labels, total):
        for pname, prange in zip(area.p_names, area.p_idx):
            l = list(prange)
            n = l[-1] - l[0]
            lticks.append(total + n/2)
            labels.append(f'{area.name}.{pname}')
            total += n
            ticks.append(total)
        return total
    total = add_area(conductor.input, ticks, lticks, labels, 0)
    for area in conductor.areas:
        total = add_area(area, ticks, lticks, labels, total)
    return ticks, lticks, labels

def get_rates(spikes, conductor, batch_size, dt, kwidth = 0):
    ticks,_,_ = get_ticks(conductor)
    rates = torch.empty(spikes.shape[0], batch_size, len(ticks)-1).to(spikes)
    for i,(lo,hi) in enumerate(zip(ticks[:-1], ticks[1:])):
        rates[:,:,i] = torch.mean(spikes[:,:,lo:hi], dim=2) / dt
    if kwidth > 0:
        if not kwidth%2:
            kwidth += 1
        kernel = torch.ones(len(ticks)-1, 1, kwidth) / kwidth
        conv_rates = torch.nn.functional.conv1d(
            rates.permute(1,2,0), kernel.to(rates),
            groups = len(ticks)-1, padding = int((kwidth-1)/2)
        ) # rates (b,npop,t) -> (b,npop,t)
        return conv_rates.permute(2,0,1) # (t,b,npop)
    else:
        return rates # (t,b,npop)
