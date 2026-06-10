#%%
from memes_base import utils
import torch
from memes_base.reddit_dataset import RedditEmbedDataset
from torch.utils.data import DataLoader

config = utils.load_config("configs/redditcfg.yaml")

raw_dset_cfg = config["Dataset"]

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%%
if __name__ == '__main__':
    ds = RedditEmbedDataset(raw_dset_cfg)

    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds),
        batch_size=10,
        drop_last=False)

    loader = DataLoader(
        ds,
        sampler=sampler)
# %%

# from transformers import CLIPProcessor, CLIPModel

# embedding_model = "openai/clip-vit-base-patch32"
# # if __name__ == '__main__':
# embed_model = CLIPModel.from_pretrained(embedding_model)
# embed_processor = CLIPProcessor.from_pretrained(embedding_model)

# print(embed_processor(["Hi my name is Jeff"], return_tensors="pt", truncation=True))

# %%

# import numpy as np
# import pyxis as px

# nb_samples = 10

# X = np.outer(np.arange(1, nb_samples + 1, dtype=np.uint8), np.arange(1, 4 + 1, dtype=np.uint8))
# y = np.arange(nb_samples, dtype=np.uint8)

# for i in range(nb_samples):
#     print('Input: {} -> Target: {}'.format(X[i], y[i]))

# #%%

# with px.Writer(dirpath='data_test', map_size_limit=20, ram_gb_limit=1) as db:
#     db.put_samples('input', X, 'target', y)
#     # db.put_samples('input', X, 'target', y)

# # %%
# with px.Reader('data_test') as db:
#     print(db)
# %%

# import lmdb
# import torch
# from safetensors.torch import save
# import pickle

# # 
# append_to_lmdb('data_test', [('key3', torch.tensor([0, 1])), ('key4', torch.tensor([2, 3]))])
# a = read_from_lmdb('data_test', ["key3", "key4"])

# list(a)

# get_keys_from_lmdb('data_test')


#%%

# from safetensors.torch import save_file, safe_open
# import torch

# data1= {'bob': torch.tensor([1, 2, 3, 4, 5])}
# data2= {'bob1': torch.tensor([6, 7, 8, 9, 10])}
# save_file(data1, 'data_test.safetensors')
# save_file(data2, 'data_test.safetensors', append=True)
# # %%
# with safe_open("data_test.safetensors", framework='pt') as f:
#     for key in f.keys():
#         print(f.get_tensor(key))
# %%
