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

def load_file(path):
    with open(path, 'r') as ymlfile:
        return Box(yaml.load(ymlfile, Loader=loader))

def load_config(master):
    if type(master) == dict:
        cfg = master
        dir = Path()
    else:
        cfg = load_file(master)
        dir = Path(master).parent
    cfg.model = load_file(dir / cfg.model_config)
    cfg.train = load_file(dir / cfg.train_config)
    cfg.tspec = Box(dict(device=torch.device('cuda:0'), dtype=torch.float))
    return cfg

def set(master):
    global cfg
    cfg = load_config(master)

set(Path(__file__).parent / 'configs/default.yaml')