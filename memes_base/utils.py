import importlib
import yaml
import os
import torch
import re

def buildFromConfig(conf, run_time_args = {}):
    """Builds an object from a yaml-style config

    Args:
        conf (dict): The config to build from. Must have keys: 'module', 'class', and 'args'
        run_time_args (dict, optional): Any additional arguments to pass to the constructor that aren't in the config. Defaults to {}.

    Raises:
        ValueError: If no module to import from is specified

    Returns:
        object: The built object
    """
    if 'module' in conf:
        module = importlib.import_module(conf['module'])
        cls = getattr(module, conf['class'])
        return cls(**conf['args'], **run_time_args)
    else:
        raise ValueError('No module specified in config.')
    
def load_config(config_path):
    """Loads a config from a .yaml file. Essentially a wrapper for yaml.load

    Args:
        config_path (string): The path to the file

    Returns:
        dict: The config dictionary
    """
    with open(config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf

def build_agent_config(base_config, wandb_config, runid):
    """Modifies a config with the hyperparameters in a W&B config given by a sweep. Works in-place

    Args:
        base_config (dict): The config to be modified
        wandb_config (dict): The W&B config containing sweep-assigned hyperparameters
        runid (string): The id of the run. Used to create the path for training saves
    """
    #* IMPORTANT: Works in-place
    base_config["Training"].update(wandb_config["Training"])
    base_config["Model"]["args"].update(wandb_config["Model"])
    base_config["Training_Directory"] += f"run_{runid}/"

def build_wandb_config(config, config_path):
    """Makes a W&B-suitable config from a specified config (presumable loaded from a file)

    Args:
        config (dict): the base config to use in making the W&B config
        config_path (string): the path to the base config (stored in W&B for future reference)

    Returns:
        dict: the W&B-compatible config
    """
    #* IMPORTANT: Doesn't work in-place
    wandb_config = {}
    wandb_config["Training"] = config["Training"]
    wandb_config["Model"] = config["Model"]["args"]
    wandb_config["base_config"] = config_path

    return wandb_config

def latest_save(save_dir, device):
    """Finds and loads the latest save of a model from the specified directory

    Args:
        save_dir (string): The directory the model saves are stored in
        device (string): The device that the model is loaded on ("cuda" or "cpu")

    Returns:
        int, dict(torch.tensor): the epoch that the loaded model is from, and the dictionary of model (and optimizer) states
    """
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