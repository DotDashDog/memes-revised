import importlib
import yaml
import os
import torch
import re

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

def build_agent_config(base_config, wandb_config, run):
    #* IMPORTANT: Works in-place
    base_config["Training"].update(wandb_config["Training"])
    base_config["Model"]["args"].update(wandb_config["Model"])
    base_config["Training_Directory"] += f"run_{run.id}/"

def build_wandb_config(config, config_path):
    #* IMPORTANT: Doesn't work in-place
    wandb_config = {}
    wandb_config["Training"] = config["Training"]
    wandb_config["Model"] = config["Model"]["args"]
    wandb_config["base_config"] = config_path

    return wandb_config

def latest_save(save_dir, device):
    files = os.listdir(save_dir)
    maxfile = None
    epoch = -1
    for file in files: #* Filename should take the form 'model_epoch_{ep}.pt'
        filename, file_ext = os.path.splitext(file)
        if file_ext != ".pt":
            continue
        
        #* Extract epoch
        current_epoch = int(re.findall(r'\d+', filename)[0])

        if current_epoch > epoch:
            epoch = current_epoch
            maxfile = file

    if maxfile is None:
        print(f"No model saves found at {save_dir}")
        checkpoint = None
    else:
        print(f"Found latest save at {os.path.join(save_dir, maxfile)}")
        checkpoint = torch.load(os.path.join(save_dir, maxfile), map_location=device)

    return epoch, checkpoint