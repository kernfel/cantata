import pytest
from box import Box
from pathlib import Path
import os
import torch
import numpy as np
from cantata import config

@pytest.fixture(scope='module')
def conf_components():
    '''
    Loads all fixture configs for modular use in test-scope fixtures,
    and ensures cfg resets after testing this module.
    '''
    original_master = config._latest_master

    path = Path(__file__).parent / 'fixtures' / 'config'
    filenames = os.listdir(path)
    bits = {}
    for f in filenames:
        base, suffix = f.split('.')
        bits[base] = config.read_file(path / f)
    yield Box(bits)

    config.load(original_master)

@pytest.fixture(params = ['cpu', 'cuda:0'])
def tspec(request):
    return Box(dict(device=torch.device(request.param), dtype=torch.float))

@pytest.fixture
def model_1(conf_components, tspec):
    conf = conf_components.base_1.copy()
    conf.model = conf_components.model_1.copy()
    conf.tspec = tspec
    config.load(conf)

@pytest.fixture
def model_1_noisy(conf_components, tspec):
    conf = conf_components.base_1.copy()
    conf.model = conf_components.model_1.copy()
    conf.model.populations.Exc1.noise_N = np.random.randint(500) + 500
    conf.model.populations.Exc1.noise_rate = np.random.rand()/conf.time_step
    conf.model.populations.Exc1.noise_weight = np.random.rand()/500
    conf.model.populations.Inh1.noise_N = np.random.randint(500) + 500
    conf.model.populations.Inh1.noise_rate = np.random.rand()/conf.time_step
    conf.model.populations.Inh1.noise_weight = np.random.rand()/500
    conf.tspec = tspec
    config.load(conf)

@pytest.fixture
def model_2(conf_components, tspec):
    conf = conf_components.base_1.copy()
    conf.model = conf_components.model_2.copy()
    conf.tspec = tspec
    config.load(conf)
