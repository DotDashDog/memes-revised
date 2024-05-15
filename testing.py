#%%
from memes_base import utils
import torch

config = utils.load_config("testcfg.yaml")

dset = utils.buildFromConfig(config["Dataset"])#, {'process_chunks' : [0, 1, 5]})
#%%

# %%
