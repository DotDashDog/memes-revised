#%%
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
#%%

# import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_urls = [
    'http://images.cocodataset.org/val2014/COCO_val2014_000000159977.jpg', 
    'http://images.cocodataset.org/val2014/COCO_val2014_000000311295.jpg',
    'http://images.cocodataset.org/val2014/COCO_val2014_000000457834.jpg', 
    'http://images.cocodataset.org/val2014/COCO_val2014_000000555472.jpg',
    'http://images.cocodataset.org/val2014/COCO_val2014_000000174070.jpg',
    'http://images.cocodataset.org/val2014/COCO_val2014_000000460929.jpg'
    ]
images = []
for url in image_urls:
    images.append(Image.open(requests.get(url, stream=True).raw))

def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

grid = image_grid(images, cols=3)
# display(grid)

#%% Zero-shot classification
classes = ['giraffe', 'zebra', 'elephant', 'teddybear', 'hotdog']
inputs = processor(text=classes, images=images, return_tensors="pt", padding=True)

# %%
outputs = model(**inputs)

#%%
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import matplotlib.pyplot as plt

# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")#"microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")#"microsoft/trocr-base-handwritten")

# # load image from the IAM dataset
# url = "https://cdn.cms-twdigitalassets.com/content/dam/blog-twitter/official/en_us/products/2022/twitter-new-edit-tweet-feature-only-test-1.jpg.img.fullhd.medium.jpg" #"https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# pixel_values = processor(image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)

# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# %%
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast, LayoutLMv3Processor

# image_processor = LayoutLMv3ImageProcessor()
# tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
# processor = LayoutLMv3Processor(image_processor, tokenizer)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

url = "https://preview.redd.it/pick-your-choice-v0-25ndxkxl2t0d1.jpeg?width=1080&crop=smart&auto=webp&s=df7bdc223ffa2f70b4e757d860aa0f96f670d831" #"https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGBA")

import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

TINT_COLOR = (0, 0, 0)
OPACITY = 100

# chars = pytesseract.image_to_string(image)
d = pytesseract.image_to_data(image, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    (llx, lly, urx, ury) = (x, y, x+w, y+h)
    overlay = Image.new('RGBA', image.size, TINT_COLOR+(0,))
    draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
    draw.rectangle(((llx, lly), (urx, ury)), fill=TINT_COLOR+(OPACITY,))

    # Alpha composite these two images together to obtain the desired result.
    image = Image.alpha_composite(image, overlay)
# encoding = processor(image, return_tensors='pt')
plt.imshow(image)
# generated_text = processor.batch_decode(encoding['input_ids'])
# %%
