import torch
import numpy as np
from cantata import cfg

def generate():
    sines = generate_sines()
    raster = generate_raster(sines)

    y_flat = torch.sum(sines, 2)/cfg.train.n_freqs
    y_data = y_flat.to(**cfg.tspec).expand(cfg.batch_size,cfg.n_steps,cfg.n_outputs)

    x_data = torch.zeros((cfg.batch_size,cfg.n_steps,cfg.n_inputs), **cfg.tspec)
    x_data[raster] = 1.0

    return x_data, y_data, sines

def generate_sines():
    freqs = torch.abs(torch.randn((1,1,cfg.train.n_freqs,1))) * cfg.train.max_freq
    phases = torch.rand((1,1,cfg.train.n_freqs,1))
    times = torch.arange(cfg.n_steps).reshape(1,-1,1,1) * cfg.time_step
    raw_rates = torch.sin(2*np.pi*(freqs * times + phases)) * .5 + .5
    return raw_rates

def generate_raster(sines):
    samplings = torch.rand((cfg.batch_size,1,cfg.train.n_freqs,cfg.n_inputs))
    rates = torch.sum(sines * samplings, 2)/cfg.train.n_freqs
    prob = rates * cfg.time_step * cfg.train.max_rate
    mask = torch.rand((cfg.batch_size,cfg.n_steps,cfg.n_inputs))
    return mask < prob
