import pytest
import cantata
from cantata import config
from box import Box
import torch
import os

@pytest.fixture(scope='module')
def dummy(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('config.dummy')
    main = Box({
        'model_config':'model.yaml',
        'train_config':'train.yaml'
    })
    model = Box({'data': 0})
    train = Box({'data': 1})
    expected = main + {
        'model': model,
        'train': train,
        'tspec': dict(device=torch.device('cuda:0'), dtype=torch.float)
    }
    main_path = tmp_path / 'main.yaml'
    model_path = tmp_path / main.model_config
    main.to_yaml(main_path)
    model.to_yaml(model_path)
    train.to_yaml(tmp_path / main.train_config)
    return Box(dict(main=main, expected=expected,
                    main_path=main_path, model_path=model_path))

@pytest.fixture(scope='module')
def scientific(tmp_path_factory):
    strings = Box({
        1: '1e2',
        2: '2e-3',
        3: '-3e4',
        4: '-4e-5',
        5: '5.6e6',
        6: '7.8e-7',
        7: '-9.1e8',
        8: '-2.3e-9'
    })
    floats = Box(dict([(k,float(v)) for k,v in strings.items()]))
    path = tmp_path_factory.mktemp('config.scientific') / 'scientific.yaml'
    with open(path, 'w') as file:
        yaml = strings.to_yaml()
        unquoted = ''.join(yaml.split("'"))
        file.write(unquoted)
    return Box(dict(path=path, floats=floats))

@pytest.fixture(scope='module')
def reset():
    original_master = config._latest_master
    yield None
    config.load(original_master)


def test_default_config_has_model():
    assert len(config.cfg.model) > 0, 'Must load sensible defaults'
def test_default_config_has_train():
    assert len(config.cfg.train) > 0, 'Must load sensible defaults'

def test_read_file(dummy):
    assert config.read_file(dummy.model_path) == dummy.expected.model

def test_read_file_scientific_notation(scientific):
    assert config.read_file(scientific.path) == scientific.floats, \
        'YML-compliant scientific notation is not read correctly'

def test_read_config_path(dummy):
    assert config.read_config(dummy.main_path) == dummy.expected
def test_read_config_master(dummy):
    os.chdir(dummy.main_path.parent)
    assert config.read_config(dummy.main) == dummy.expected

def test_load_path(dummy, reset):
    config.load(dummy.main_path)
    assert config.cfg == dummy.expected
def test_load_master(dummy, reset):
    os.chdir(dummy.main_path.parent)
    config.load(dummy.main)
    assert config.cfg == dummy.expected

def test_load_affects_references(dummy, reset):
    config.load(dummy.main_path)
    assert cantata.cfg == dummy.expected

def test_reload(dummy, reset):
    config.load(dummy.main_path)
    dummy.expected.model.data = 'A new value'
    dummy.expected.model.to_yaml(dummy.model_path)
    config.reload()
    assert config.cfg == dummy.expected
