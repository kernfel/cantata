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
    '''
    Reads the YAML file specified by @arg path and returns its contents
    in a Box.
    '''
    with open(path, 'r') as ymlfile:
        return Box(yaml.load(ymlfile, Loader=loader))

def read_config(master):
    '''
    Loads a complete configuration from an orchestrating master.
    @arg master: path_like | dict | Box
        Top-level orchestration data. Model and training configurations can be
        supplied directly in 'model' and 'train' entries, or by referring to
        subordinate files in the 'model_config' and 'train_config' entries, respectively.
        In that case, existing 'model' or 'train' entries will be overwritten.
        If a 'tspec' entry is not present, it will be populated with sensible
        defaults (device=cuda:0, dtype=float).
        * If @arg master is path_like, data is read from the indicated YAML file,
        and subordinate file paths are resolved from its parent directory.
        * Otherwise, subordinate file paths are resolved from the working directory.
    @return A Box containing the loaded configuration.
    '''
    if type(master) == dict or type(master) == Box:
        conf = Box(master)
        dir = Path()
    else:
        conf = read_file(master)
        dir = Path(master).parent
    if 'model_config' in conf:
        conf.model = read_file(dir / conf.model_config)
    if 'train_config' in conf:
        conf.train = read_file(dir / conf.train_config)
    if 'tspec' not in conf:
        conf.tspec = Box(dict(device=torch.device('cuda:0'), dtype=torch.float))
    return conf

def load(master):
    '''
    Loads a master as specified in `config.read_config` and sets the global
    `config.cfg` to the result.
    '''
    global _latest_master
    cfg.clear()
    cfg.update(read_config(master))
    _latest_master = master

def reload():
    '''
    Reloads the last master passed to `load` to e.g. reflect changes in
    configuration files.
    '''
    load(_latest_master)

cfg = Box()
load(Path(__file__).parent / 'configs' / 'base.yaml')
