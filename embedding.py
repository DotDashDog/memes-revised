#%%
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
from memes_base.image_processing import load_from_url, image_to_tensor
import cv2
import time
    

#%%


# image_urls = [
#     "https://preview.redd.it/yeah-no-im-outta-here-v0-pbykjd795nbe1.png?auto=webp&s=7870e4ae9ae77d71b30dacbe0a9bb368afde30cc", 
#     "https://preview.redd.it/dang-they-know-all-the-teaaaa-v0-212n1xypmibe1.png?auto=webp&s=7f7e1f7eb711c63519dcf755175b06dcb2051972"
# ]

# texts = [
#     "Yeah no I'm outta here",
#     "Dang they know all the teaaaa"
# ]

# images = [load_from_url(url) for url in image_urls]

# img_tensors = [image_to_tensor(img) for img in images]

# #%%
# starttime = time.time()
# embeds = embed_text_and_images(texts, img_tensors, embed_model, embed_processor)

# print(f"Embedding {len(texts)} memes took {time.time()-starttime} seconds")
#%%

# image_urls = [
#     'http://images.cocodataset.org/val2014/COCO_val2014_000000159977.jpg', 
#     'http://images.cocodataset.org/val2014/COCO_val2014_000000311295.jpg',
#     'http://images.cocodataset.org/val2014/COCO_val2014_000000457834.jpg', 
#     'http://images.cocodataset.org/val2014/COCO_val2014_000000555472.jpg',
#     'http://images.cocodataset.org/val2014/COCO_val2014_000000174070.jpg',
#     'http://images.cocodataset.org/val2014/COCO_val2014_000000460929.jpg'
#     ]
# images = []
# for url in image_urls:
#     images.append(Image.open(requests.get(url, stream=True).raw))

# def image_grid(imgs, cols):
#     rows = (len(imgs) + cols - 1) // cols
#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
    
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i%cols*w, i//cols*h))
#     return grid

# grid = image_grid(images, cols=3)