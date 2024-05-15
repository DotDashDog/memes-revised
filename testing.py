#%%
from memes_base import utils
import torch

config = utils.load_config("testcfg.yaml")

dset = utils.buildFromConfig(config["Dataset"])
#%%

# %%
