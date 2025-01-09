#%%
import cv2
import pytesseract
# from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import time
import torch

def image_to_tensor(image):
    # Convert the image to a PyTorch tensor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image_tensor = torch.from_numpy(image).float()  # Convert to float tensor

    return image_tensor

def load_from_url(image_url):
    # Download the image from the URL
    response = requests.get(image_url, stream=True)
    response.raise_for_status()  # Ensure the request was successful
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


def preprocess_1(image):
    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply thresholding

    return thresh

def preprocess_2(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return img

def preprocess_3(im):
    #* The best preprocessing method so far
    #! STILL NEED TO OPTIMIZE PREPROCESSING
    # im= cv2.bilateralFilter(img,5, 55,60)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 235, 255, 1)

    return im


def extract_text_from_image(image, backend='tesseract', preprocess_fn=preprocess_3):

    #* EasyOCR (deep-learning based) is much (~10-20x) slower than Tesseract, but more accurate.
    valid_backends = ['tesseract', 'easyocr']

    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend}. Must be one of {valid_backends}")

    processed_image = preprocess_fn(image)

    # Perform OCR
    #! Figure out a good way to differentiate different text blocks
    if backend == 'tesseract':
        text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 11')
    elif backend == 'easyocr':
        reader = easyocr.Reader(['en'])
        result = reader.readtext(processed_image)
        text = " ".join([text[1] for text in result])

    return text


#%%
# url = "https://preview.redd.it/yeah-no-im-outta-here-v0-pbykjd795nbe1.png?auto=webp&s=7870e4ae9ae77d71b30dacbe0a9bb368afde30cc"

#"https://preview.redd.it/dang-they-know-all-the-teaaaa-v0-212n1xypmibe1.png?auto=webp&s=7f7e1f7eb711c63519dcf755175b06dcb2051972"
#"https://preview.redd.it/this-is-sum-bullsh-v0-uvjqmncgambe1.jpeg?auto=webp&s=568ee0f66bfe5ef0c984a619fc96ff2a7af5f804"
#"https://preview.redd.it/anybody-else-do-this-v0-0chsp9xzkmbe1.png?auto=webp&s=849b9642b8a484774a6cdd49d25ab676713894c6"
#"https://preview.redd.it/td211nt8tlbe1.png?width=1080&crop=smart&auto=webp&s=e3b5372880cbfa512e9fe70223d1c6b7042e0528"
#"https://preview.redd.it/ic9kj8aw0kbe1.jpeg?auto=webp&s=66a49fb8bea6f2e6b3659c06ee3165a71313877d"
#"https://preview.redd.it/couch-sleepy-not-the-same-as-bed-sleepy-v0-hpcvqgnazibe1.jpeg?auto=webp&s=e45671571dd67146d5e95bd07d407af8c27bd935"
#"https://preview.redd.it/pov-people-who-dont-know-about-the-japanese-justice-systems-v0-h8supju4ombe1.jpeg?auto=webp&s=6d7a4b7ef19ef68b4d0ee7b9500e3b198b4df046"
#"https://preview.redd.it/based-on-a-true-story-v0-bjxgcxssqmbe1.jpeg?width=1080&crop=smart&auto=webp&s=61e3ad621be36312b602ab362defff6a10a38522"
#"https://external-preview.redd.it/google-if-it-pays-me-ill-allow-it-v0-qDok1rlqGxVwE32X3QAlm51S_IrATHG1XQoRmNxPqz4.png?width=1080&crop=smart&auto=webp&s=a6ce94c304a3305eab447d634877c60f08a1e1a1"
#"https://preview.redd.it/pick-your-choice-v0-25ndxkxl2t0d1.jpeg?width=1080&crop=smart&auto=webp&s=df7bdc223ffa2f70b4e757d860aa0f96f670d831" #"https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"

# image = load_from_url(url)

# starttime = time.time()
# extracted_text = extract_text_from_image(image, backend='tesseract')

# print(f"In {time.time()-starttime} Extracted Text With Tesseract:")
# print(extracted_text)

# starttime = time.time()
# extracted_text = extract_text_from_image(image, backend='easyocr')

# print(f"In {time.time()-starttime} Extracted Text With EasyOCR:")
# print(extracted_text)
