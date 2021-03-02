import pytest
from box import Box
from pathlib import Path
import os
import torch
import numpy as np
from cantata import config

@pytest.fixture(scope='session')
def conf_components():
    '''
    Loads all fixture configs for modular use in test-scope fixtures,
    and ensures cfg resets after testing this module.
    '''
    path = Path(__file__).parent / 'fixtures' / 'config'
    filenames = os.listdir(path)
    bits = {}
    for f in filenames:
        base, suffix = f.split('.')
        bits[base] = config.read_file(path / f)
    return Box(bits)

@pytest.fixture
def rconf():
    return Box(dict(batch_size = 4, dt = 1e-3))

@pytest.fixture
def model1(conf_components):
    return config.read(conf_components.model1)

@pytest.fixture
def model2(conf_components):
    return config.read(conf_components.model2)

@pytest.fixture
def model1_noisy(conf_components, rconf):
    conf = conf_components.model1.copy()
    conf.areas.A1.populations.Exc.noise_N = np.random.randint(500) + 500
    conf.areas.A1.populations.Exc.noise_rate = np.random.rand()/rconf.dt
    conf.areas.A1.populations.Exc.noise_weight = np.random.rand()/500
    conf.areas.A1.populations.Inh.noise_N = np.random.randint(500) + 500
    conf.areas.A1.populations.Inh.noise_rate = np.random.rand()/rconf.dt
    conf.areas.A1.populations.Inh.noise_weight = np.random.rand()/500
    return config.read(conf)
