#%%
from memes_base import utils
import torch

config = utils.load_config("configs/redditcfg.yaml")

dset = utils.buildFromConfig(config["Dataset"])
#%%

# %%
