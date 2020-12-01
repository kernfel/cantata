import pytest
from box import Box
from pathlib import Path
import os
import torch
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

@pytest.fixture
def model_1(conf_components):
    conf = conf_components.base_1.copy()
    conf.model = conf_components.model_1.copy()
    conf.tspec = Box(dict(device=torch.device('cpu'), dtype=torch.float))
    config.load(conf)

@pytest.fixture
def model_2(conf_components):
    conf = conf_components.base_1.copy()
    conf.model = conf_components.model_2.copy()
    conf.tspec = Box(dict(device=torch.device('cpu'), dtype=torch.float))
    config.load(conf)
