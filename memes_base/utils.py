import importlib
import yaml

def buildFromConfig(conf, run_time_args = {}):
    if 'module' in conf:
        module = importlib.import_module(conf['module'])
        cls = getattr(module, conf['class'])
        return cls(**conf['args'], **run_time_args)
    else:
        raise ValueError('No module specified in config.')
    
def load_config(config_path):
    with open(config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf