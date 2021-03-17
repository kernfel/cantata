import torch
import numpy as np
import scipy.stats
import cantata
import cantata.analysis.util as util

def quantify_unstimulated(model, dt, early = (10,20), late = (50,60)):
    '''
    @arg model: Conductor or compatible model instance
    @arg early: Early observation period, in integer seconds
    @arg late: Late observation period, in integer seconds
    @return a 4-tuple (r0, w0, r1, w1), where
        - r0, r1 are rate measures, see get_rate_measures()
        - w0, w1 are STDP measures, see get_stdp_measures()
        - *0, *1 refer to early and late measures, respectively.
    '''
    STDP = model.areas[0].synapses_int.longterm
    mask, *midx = util.get_STDP_mask(model.areas[0], 'Exc', 'Exc')
    wmax = model.areas[0].synapses_int.wmax[mask[0]][0]
    batch_size = STDP.W.shape[0]
    def get_empty(secs):
        return util.get_inputs(int(np.round(secs/dt)), (0,), (0,),
                               device = mask.device, batch_size = batch_size)

    # Settle without observation until early begins:
    if early[0] > 0:
        model(get_empty(early[0]))

    # Observe spikes, STDP in early window:
    Wpre = STDP.W[mask]
    X = model(get_empty(early[1] - early[0]))
    Wpost = STDP.W[mask]
    w0 = get_stdp_measures(torch.stack((Wpre, Wpost)) / wmax, midx)
    r0 = get_rate_measures(X[1], model.areas[0], dt)

    # Run through to late window:
    model(get_empty(late[0] - early[1]))

    # Observe spikes, STDP in late window:
    Wpre = STDP.W[mask]
    X = model(get_empty(late[1] - late[0]))
    Wpost = STDP.W[mask]
    w1 = get_stdp_measures(torch.stack((Wpre, Wpost)) / wmax, midx, t=-1)
    r1 = get_rate_measures(X[1], model.areas[0], dt)

    return r0, w0, r1, w1

def quantify_stimulated(model, periods, rates, dt, onset = 20,
                        settle = 5, early = (10,15), late = (55,60)):
    '''
    @arg model: Conductor or compatible model instance
    @arg periods: Stimulus durations, in ticks
    @arg rates: Stimulus levels, in Hz
    @arg onset: Duration of stimulus onset (for onset-sensitive unit detection),
        in ticks
    @arg settle: Unstimulated initial period, in integer seconds
    @arg early: Early observation period, in integer seconds
    @arg late: Late observation period, in integer seconds
    @return a 6-tuple (r0, w0, s0, r1, w1, s1), where
        - r0, r1 are rate measures, see get_rate_measures()
        - w0, w1 are STDP measures, see get_stdp_measures()
        - s0, s1 are stimulated measures, see get_stimulated_measures()
        - *0, *1 refer to early and late measures, respectively.
    '''
    assert int(1/dt) % sum(periods) == 0,\
        'Periods must tile neatly into 1-second segments.'
    for p in periods:
        assert p > onset, 'Periods must be longer than the onset'
    assert len(periods) == 2, 'Multi-level stimulations NYI'
    STDP = model.areas[0].synapses_int.longterm
    mask, *midx = util.get_STDP_mask(model.areas[0], 'Exc', 'Exc')
    wmax = model.areas[0].synapses_int.wmax[mask[0]][0]
    batch_size, device = STDP.W.shape[0], STDP.W.device

    # Settle without observation and without stimulation:
    if settle > 0:
        model(util.get_inputs(int(settle/dt), (0,), (0,),
                              device = device, batch_size = batch_size))

    inputs = util.get_inputs(int(late[1]/dt), periods, rates,
                             batch_size = batch_size)

    # Run without observation until early begins:
    t = 0
    if early[0] > 0:
        t = int(early[0]/dt)
        model(inputs[:t].to(device))

    # Observe spikes, STDP in early window:
    t0, t = t, int(early[1]/dt)
    Wpre = STDP.W[mask]
    X = model(inputs[t0:t].to(device))
    Wpost = STDP.W[mask]
    w0 = get_stdp_measures(torch.stack((Wpre, Wpost)) / wmax, midx)
    r0 = get_rate_measures(X[1], model.areas[0], dt)
    s0 = get_stimulated_measures(X[1], model.areas[0], periods, dt, onset)

    # Run through to late window:
    t0, t = t, int(late[0]/dt)
    model(inputs[t0:t].to(device))

    # Observe spikes, STDP in late window:
    t0, t = t, int(late[1]/dt)
    Wpre = STDP.W[mask]
    X = model(inputs[t0:t].to(device))
    Wpost = STDP.W[mask]
    w1 = get_stdp_measures(torch.stack((Wpre, Wpost)) / wmax, midx, t=-1)
    r1 = get_rate_measures(X[1], model.areas[0], dt)
    s1 = get_stimulated_measures(X[1], model.areas[0], periods, dt, onset)

    return r0, w0, s0, r1, w1, s1

def get_rate_measures(X, area, dt, quantiles = torch.arange(0,1.1,.1)):
    '''
    @arg X: (t,b,N) in area
    @returns (npops, 4 + |quantiles|) tensor containing:
        - Rate, grand mean and stddev
        - Variance over time of instantaneous rate, batch mean and stddev
        - Rate quantiles
    '''
    rates = X.mean(dim=(0,1)) / dt # Hz, (N)
    ret = torch.zeros(len(area.p_idx), len(quantiles)+4)
    for i, idx in enumerate(area.p_idx):
        rstd, rmean = torch.std_mean(rates[idx])
        q = torch.quantile(rates[idx], quantiles.to(rates)).cpu()
        vstd, vmean = torch.std_mean(X[:,:,idx].mean(dim=2).var(dim=0))
        ret[i] = torch.cat((torch.tensor([rmean, rstd, vmean, vstd]), q))
    return ret.cpu()

def get_stdp_measures(W, midx, t = 0,
                      quantiles = torch.arange(0,1.1,.1), tol = 1e-4):
    '''
    @arg W: (T,masked); a masked STDP.W at two or more time points, normalised
        to the connection's wmax
    @arg midx: A 3-tuple (iBatch,iPre,iPost) of the respective indices for W
    @returns (6 + |quantiles|) tensor containing, in order:
        - Cosine similarity across time: Mean, stddev
            Note, comparison is between t=0 and t=-1 only.
            Mean and std here are calculated across the batch.
        - Weight values at time t: Mean, stddev
        - Proportion of weights at lower, upper bound (+- tol) at time t
        - Weight quantiles at time t
    '''
    q = torch.quantile(W[t], quantiles.to(W)).cpu()
    std, mean = torch.std_mean(W[t])

    # Recall that midx[0] is mask-selected batch indices.
    # Since cantata.init.build_connectivity() guarantees a fixed number of conns
    # per projection, the final element of midx[0] must be batch_size-1.
    batch_size = midx[0][-1] + 1
    # Furthermore, for the same reason, W[t] can be split up into evenly sized
    # chunks corresponding to flattened (pre,post) at a given batch index.
    W = W.view(W.shape[0], batch_size, -1) # (t,b,masked)

    # Cosine similarity between weight vectors at start & end of period,
    # mean & std:
    cs = torch.nn.functional.cosine_similarity(W[0], W[-1], dim=-1)
    time_std, time_mean = torch.std_mean(cs)

    # Proportion of saturated weights at higher/lower weight bound
    sat_norm = W.shape[1] * W.shape[2]
    sat_high = torch.sum(W[t] > 1-tol)/sat_norm
    sat_low = torch.sum(W[t] < tol)/sat_norm

    ret = torch.tensor([time_mean, time_std, mean, std, sat_low, sat_high])
    return torch.cat((ret, q))

def get_stimulated_measures(X, area, periods, dt, onset = 20,
                            quantiles = torch.arange(0,1.1,.1), sig=.05):
    '''
    Given a spike train of the model under stimulation, provides a number of
    statistical descriptions of the data.
    @arg X: (t,b,N) spikes in area
    @arg area: SNN-compatible model area instance
    @arg periods: iterable of pulse durations in ticks
    @arg onset: Duration of pulse onset in ticks
    @return (npops, nperiods, |quantiles|+4) tensor, containing:
        - Proportion of level-sensitive neurons, i.e., neurons that
            significantly increase their firing rate during the stimulus,
            compared to other periods
        - Proportion of onset-sensitive neurons, i.e., neurons that
            significantly change their firing rate (in either direction)
            during stimulus onset, compared to the full stimulus duration
        - Firing rates in the complete pulse, mean and stddev across neurons
        - Firing rate quantiles in the complete pulse
    '''
    stop, total = 0, int(sum(periods))
    T, batch_size, N = X.shape
    nperiods, nreps = len(periods), T // total
    assert total*nreps == T, 'Periods must neatly tile the input.'

    r_pulse = torch.empty(nperiods, nreps, batch_size, N)
    r_onset = torch.empty(nperiods, nreps, batch_size, N)
    for i,p in enumerate(periods):
        p = int(p)
        idx = torch.arange(stop, stop+p).expand(nreps,-1) \
            + (torch.arange(nreps)*total)[:,None] # (nreps, p)
        S = X[idx,:,:] # (nreps, p, b, N)
        r_pulse[i] = S.sum(dim=1).cpu() / (p * dt) # Hz, (nreps, b, N)
        r_onset[i] = S[:,:onset].sum(dim=1).cpu() / (onset * dt) # Hz
        stop += p

    ret = torch.empty(len(area.p_idx), nperiods, len(quantiles)+4)
    for i,idx in enumerate(area.p_idx):
        rx_pulse = r_pulse[:,:,:,idx] # (nperiods, nreps, b, Nx)
        rx_onset = r_onset[:,:,:,idx]

        # Firing rates during each stimulus (full pulse duration)
        r = rx_pulse.mean(dim=1) # Hz, (nperiods, b, Nx)
        std_pulse, mean_pulse = torch.std_mean(r, dim=(1,2)) # (nperiods)
        q_pulse = np.quantile(
            r.numpy(), quantiles.numpy(), axis=(1,2)) # (|q|, nperiods)
        q_pulse = torch.tensor(q_pulse, dtype=torch.float)

        # Finding pulse level sensitive units
        # 2-tailed independent test; assumes that on and off are independent,
        # even though they follow each other
        # Value reflects the fraction of units that significantly increase their
        # firing for a given stimulus.
        level_sensitive = torch.empty(nperiods) # (nperiods)
        for j in range(nperiods):
            if nperiods == 2:
                rx_off = rx_pulse[(j+1) % 2]
            else:
                raise NotImplementedError
            _, p = scipy.stats.ttest_ind(
                rx_pulse[j].numpy(), rx_off.numpy(),
                axis=0, alternative='greater') # (b, Nx)
            level_sensitive[j] = np.sum(p<sig) / (batch_size*N) # scalar

        # Finding onset-selective units
        # 2-tailed paired test, since onset and rest-of-pulse are tightly linked
        _, p = scipy.stats.ttest_rel(
            rx_onset, rx_pulse, axis=1) # (nperiods, b, Nx)
        onset_sensitive = np.sum(p<sig, axis=(1,2)) / (batch_size*N) # (nperiods)
        onset_sensitive = torch.tensor(onset_sensitive, dtype=torch.float)

        ret[i] = torch.cat((
            level_sensitive[:,None],
            onset_sensitive[:,None],
            mean_pulse[:,None],
            std_pulse[:,None],
            q_pulse.T
        ), dim=1)
    return ret # (nareas, nperiods, |q|+4)
