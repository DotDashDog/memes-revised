# #%%
from memes_base import utils
import torch
from memes_base.reddit_dataset import RedditEmbedDataset

config = utils.load_config("configs/redditcfg.yaml")

raw_dset_cfg = config["Dataset"]
#%%
if __name__ == '__main__':
    ds = RedditEmbedDataset(raw_dset_cfg)
# %%

# from transformers import CLIPProcessor, CLIPModel

# embedding_model = "openai/clip-vit-base-patch32"
# if __name__ == '__main__':
#     embed_model = CLIPModel.from_pretrained(embedding_model)
#     embed_processor = CLIPProcessor.from_pretrained(embedding_model)

# print(embed_processor(["Hi my name is Jeff"], return_tensors="pt", truncation=True))

# %%
