import yaml
import re
from box import Box
from pathlib import Path
import torch

# Parse scientific notation correctly
# cf. https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
# cf. https://github.com/yaml/pyyaml/issues/173
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def read_file(path):
    with open(path, 'r') as ymlfile:
        return Box(yaml.load(ymlfile, Loader=loader))

def read_config(master):
    if type(master) == dict:
        conf = master
        dir = Path()
    else:
        conf = read_file(master)
        dir = Path(master).parent
    conf.model = read_file(dir / conf.model_config)
    conf.train = read_file(dir / conf.train_config)
    conf.tspec = Box(dict(device=torch.device('cuda:0'), dtype=torch.float))
    return conf

def load(master):
    global cfg, _latest_master
    cfg = read_config(master)
    _latest_master = master

def reload():
    conf = read_config(_latest_master)
    cfg.update(conf)


load(Path(__file__).parent / 'configs/default.yaml')
