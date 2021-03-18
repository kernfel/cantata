import cantata
import torch
import numpy as np
import inspect
import pickle
import os

def runonce(func, path, rerun = False):
    '''
    If `path` is present and `rerun` is False:
        Loads and returns results from `path`.
    Otherwise:
        Runs `func`, saves the results to `path`, and returns them.
    If `func` is a generator function (i.e., yields rather than returning),
        each yielded result is saved in turn, overwriting preceding results.
    CAUTION! Always yield incremental results; older results are discarded!
    '''
    if not rerun:
        try:
            with open(path, 'rb') as file:
                result = pickle.load(file)
                print('Successfully loaded result from ' + path + '.')
                return result
        except:
            pass

    print('Running ' + func.__name__ + ', saving to ' + path + '.')
    if inspect.isgeneratorfunction(func):
        for i, result in enumerate(func()):
            if i > 0:
                os.replace(path, path + '.1')
            with open(path, 'wb') as file:
                pickle.dump(result, file)
        try:
            os.remove(path + '.1')
        except:
            pass
    else:
        result = func()
        with open(path, 'wb') as file:
            pickle.dump(result, file)

    return result

def get_inputs(n_steps, periods, rates,
               device = torch.device('cpu'), batch_size = None):
    '''
    Expects n_steps, periods in ticks.
    Returns (nsteps,) tensor, if batch_size==None (default),
        or (nsteps, batch_size, 1), if a batch_size is given.
    '''
    seq = torch.arange(n_steps, device=device)
    pattern = torch.zeros(n_steps, device=device)
    stop, total = 0, int(sum(periods))
    for rate, period in zip(rates, periods):
        pattern = torch.where(
            seq % total > stop,
            torch.tensor([rate],device=device,dtype=torch.float),
            pattern
        )
        stop += int(period)
    if batch_size is not None:
        return pattern.reshape(-1,1,1).expand(-1, batch_size, 1)
    else:
        return pattern

def get_STDP_mask(area, name_pre, name_post):
    '''
    @arg area: An SNN or compatible instance
    @arg name_pre, name_post: String names of the pre and post populations.
    Currently only supports area-internal connections.
    @return (batch, pre, post) boolean mask into the STDP weight matrix
    @return (|W[mask]|) batch, pre, and post indices to identify masked results
    '''
    synapse = area.synapses_int
    b,e,o = synapse.longterm.W.shape
    ipre = area.p_idx[area.p_names.index(name_pre)]
    ipost = area.p_idx[area.p_names.index(name_post)]

    projection = torch.zeros(e,o, dtype=torch.bool) # (e,o)
    projection[np.ix_(ipre, ipost)] = True

    active = synapse.signs != 0 # ([b],e,o)

    plastic = (synapse.longterm.A_p != 0) * (synapse.longterm.A_d != 0) # (e,o)

    mask = active * plastic * projection.to(active) # ([b],e,o)
    mask = mask.expand_as(synapse.longterm.W) # (b,e,o)

    batch = torch.arange(b)[:,None,None].expand_as(mask).to(mask)[mask]
    pre = torch.arange(e)[None,:,None].expand_as(mask).to(mask)[mask]
    post = torch.arange(o)[None,None,:].expand_as(mask).to(mask)[mask]
    return mask, batch, pre, post
