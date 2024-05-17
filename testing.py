#%%
from memes_base import utils
import torch
# import progressbar
# import time
# import random

# epochs = 100

# epoch_widgets = [
#         ' [Epoch: ', progressbar.Counter(format='%3d'), f'/{epochs}, ',
#         progressbar.Timer(), '] ',
#         progressbar.Bar('█', left='|', right='|'),
#         ' (', progressbar.AdaptiveETA(), ') '
#         ]

# epoch_progress = progressbar.ProgressBar(maxval=epochs, widgets=epoch_widgets).start()

# for i in range(epochs):
#     time.sleep(0.2)
#     epoch_progress.update(i)

# epoch_progress.finish()

config = utils.load_config("testcfg.yaml")

dset = utils.buildFromConfig(config["Dataset"])#, {'process_chunks' : [0, 1, 5]})
#%%

# %%
