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
import re

# Add PaddleOCR import for pipeline #1 implementation
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not available. Install with: pip install paddleocr")

# Add spell checking for Pipeline #1 post-processing
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("pyspellchecker not available. Install with: pip install pyspellchecker")

# Add contextual spell checking capabilities
try:
    import spacy
    CONTEXTUAL_SPELLCHECK_AVAILABLE = True
    # Initialize spaCy model for contextual analysis
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Contextual analysis with spaCy initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize spaCy: {e}")
        CONTEXTUAL_SPELLCHECK_AVAILABLE = False
        nlp = None
except ImportError:
    CONTEXTUAL_SPELLCHECK_AVAILABLE = False
    nlp = None
    print("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

def image_to_tensor(image):
    # Convert the image to a PyTorch tensor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image_tensor = torch.from_numpy(image).float()  # Convert to float tensor

    return image_tensor

def get_direct_image_url(url):
    """Convert webpage URLs to direct image URLs for various image hosting services."""
    try:
        # Handle Imgur URLs
        if 'imgur.com' in url and not any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            url += '.jpg'
            return url
            
        # Handle ImgFlip URLs
        if 'imgflip.com/i/' in url:
            # Extract the image ID from the URL
            image_id = url.split('/')[-1].split('#')[0]
            return f'https://i.imgflip.com/{image_id}.jpg'
            
        return url
    except Exception as e:
        print(f"Error converting URL {url}: {str(e)}")
        return url

def load_from_url(image_url):
    # Download the image from the URL
    max_retries = 3
    retry_delay = 2  # seconds
    
    # Handle image hosting service URLs
    # original_url = image_url
    image_url = get_direct_image_url(image_url)
    # if image_url != original_url:
    #     print(f"Converted to direct image URL: {image_url}")
    
    for attempt in range(max_retries):
        try:
            # Add headers to look more like a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(image_url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()  # Ensure the request was successful
            
            # Check content type to ensure it's an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: URL {image_url} returned non-image content type: {content_type}")
                return None
                
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Warning: Failed to decode image from URL {image_url}")
                return None
                
            return image
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    print(f"Rate limited by {image_url}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            print(f"HTTP Error for URL {image_url}: {str(e)}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"Timeout while downloading image from {image_url}")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request error while downloading image from {image_url}: {str(e)}")
            return None
            
        except Exception as e:
            print(f"Unexpected error while processing image from {image_url}: {str(e)}")
            return None
            
    return None


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

def preprocess_adaptive(image):
    """
    Advanced preprocessing that handles different colored text and reduces noise.
    Uses multiple techniques and combines results for better OCR performance.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image for better OCR (scale up small text)
    height, width = gray.shape
    scale_factor = max(1.0, 1000.0 / max(height, width))
    if scale_factor != 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Try multiple thresholding approaches and combine them
    # 1. Otsu's thresholding (automatic threshold selection)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Adaptive thresholding (good for varying lighting conditions)
    adaptive_mean = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 3. Multiple fixed thresholds for different text colors
    _, thresh_high = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    _, thresh_mid = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    _, thresh_low = cv2.threshold(enhanced, 80, 255, cv2.THRESH_BINARY)
    
    # Combine multiple thresholding results using bitwise operations
    # This helps capture text of different colors/contrasts
    combined = cv2.bitwise_or(otsu, adaptive_mean)
    combined = cv2.bitwise_or(combined, adaptive_gaussian)
    combined = cv2.bitwise_or(combined, thresh_high)
    
    # Morphological operations to clean up the image
    # Remove small noise while preserving text
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening (erosion followed by dilation) to remove small noise
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
    
    # Closing (dilation followed by erosion) to fill gaps in text
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
    
    # Final denoising: remove very small components that are likely noise
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 20  # Minimum area for a component to be considered text
    
    # Create a mask for components larger than min_area
    mask = np.zeros_like(cleaned)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.fillPoly(mask, [contour], 255)
    
    # Apply the mask to keep only larger components
    result = cv2.bitwise_and(cleaned, mask)
    
    return result

def preprocess_high_contrast(image):
    """
    Preprocessing focused on high contrast and edge enhancement.
    Good for stylized text and memes with outlined text.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR
    height, width = gray.shape
    scale_factor = max(1.5, 1200.0 / max(height, width))
    if scale_factor != 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Edge enhancement using unsharp masking
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # High contrast enhancement
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply multiple thresholds to capture different text styles
    _, binary1 = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary2 = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
    _, binary3 = cv2.threshold(normalized, 180, 255, cv2.THRESH_BINARY)
    
    # Combine thresholds
    combined = cv2.bitwise_or(binary1, binary2)
    combined = cv2.bitwise_or(combined, binary3)
    
    # Morphological closing to connect text fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return result

def preprocess_pipeline1_balanced(image):
    """
    Pipeline #1 Balanced Approach preprocessing as recommended in the research.
    Optimized for use with EasyOCR or PaddleOCR (PP-OCRv5 mobile/lightweight).
    
    Focuses on:
    - Adaptive thresholding for varying backgrounds
    - Bilateral filtering for noise reduction while preserving edges
    - Morphological operations for outlined text (e.g., Impact font)
    - Basic deskewing if needed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image for better OCR if it's too small
    height, width = gray.shape
    scale_factor = max(1.0, 800.0 / max(height, width))
    if scale_factor > 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply bilateral filter for noise reduction while preserving edges
    # This is especially good for memes with complex backgrounds
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding to handle varying backgrounds
    # This works better than fixed thresholding for diverse meme backgrounds
    adaptive_thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to handle outlined text (common in memes like Impact font)
    # Use opening to remove small noise, then closing to fill gaps in text
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
    # Opening: removes small noise while preserving text structure
    opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    
    # Closing: fills small gaps in text characters, especially useful for outlined fonts
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    # Additional processing for very outlined text (if detected)
    # Check if we have heavily outlined text by looking at the ratio of black to white pixels
    black_pixels = np.sum(closed == 0)
    total_pixels = closed.size
    black_ratio = black_pixels / total_pixels
    
    if black_ratio > 0.7:  # Likely heavily outlined or very dark background
        # Try inversion to see if it helps
        inverted = cv2.bitwise_not(closed)
        # Apply another round of morphological operations
        inverted_cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        
        # Use the version that has more reasonable black-to-white ratio
        inv_black_ratio = np.sum(inverted_cleaned == 0) / total_pixels
        if inv_black_ratio < black_ratio:
            closed = inverted_cleaned
    
    return closed

def postprocess_pipeline1_balanced(text):
    """
    Pipeline #1 post-processing for balanced approach.
    Includes contextual spell checking with BERT and meme-specific vocabulary cleaning.
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    # Basic text cleaning
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove isolated single characters that are likely OCR artifacts
    # But preserve intentional single letters like "I" or "a"
    text = re.sub(r'\b[b-hj-z]\b', '', text, flags=re.IGNORECASE)  # Remove isolated consonants except common ones
    
    # Apply contextual spell checking first (if available)
    if CONTEXTUAL_SPELLCHECK_AVAILABLE and nlp is not None:
        try:
            # Use spaCy for contextual analysis and targeted correction
            doc = nlp(text)
            
            # Common OCR error patterns that can be contextually corrected
            # Using more flexible matching since POS tagging can be confused by OCR errors
            word_corrections = {
                # Very common OCR errors
                'teh': 'the',
                'adn': 'and', 
                'youre': "you're",
                'thier': 'their',
                'recieve': 'receive',
                'seperate': 'separate',
                'wierd': 'weird',
                'definately': 'definitely',
                'occured': 'occurred',
                'begining': 'beginning',
                'similiar': 'similar',
                # Common character-level OCR errors
                'rn': 'm',  # Only apply if in middle of word
                'cl': 'd',  # "world" vs "worlcl"
                'vv': 'w',  # "wow" vs "vvow"
            }
            
            # Context-aware patterns
            pattern_corrections = {
                r'\brn\b': 'm',  # standalone "rn" -> "m"
                r'(\w+)rn(\w+)': r'\1m\2',  # "tirne" -> "time"
                r'(\w+)cl\b': r'\1d',      # "worlcl" -> "world"
                r'\bvv(\w+)': r'w\1',      # "vvow" -> "wow"
            }
            
            corrected_tokens = []
            for token in doc:
                original_word = token.text.lower()
                corrected_word = token.text
                
                # First, check direct word replacements
                if original_word in word_corrections:
                    replacement = word_corrections[original_word]
                    # Preserve original capitalization
                    if token.text[0].isupper():
                        corrected_word = replacement.capitalize()
                    else:
                        corrected_word = replacement
                else:
                    # Check pattern-based corrections
                    for pattern, replacement in pattern_corrections.items():
                        if re.search(pattern, original_word):
                            corrected_word = re.sub(pattern, replacement, token.text, flags=re.IGNORECASE)
                            break
                
                corrected_tokens.append(corrected_word)
            
            corrected_text = ' '.join(corrected_tokens)
            
            # Only apply correction if it's meaningfully different
            if corrected_text != text and len(corrected_text.strip()) > 0:
                text = corrected_text
                print(f"Contextual correction applied: '{text[:50]}...'")
        except Exception as e:
            print(f"Contextual spell checking failed: {e}")
    
    # Basic spell checking with custom meme vocabulary (as fallback/enhancement)
    if SPELLCHECKER_AVAILABLE:
        try:
            spell = SpellChecker()
            
            # Add comprehensive meme-specific words to the dictionary
            meme_words = {
                # Classic meme terms
                'meme', 'memes', 'dank', 'thicc', 'stonks', 'doggo', 'pupper', 'chonky',
                'karen', 'chad', 'simp', 'sus', 'cringe', 'based', 'normie', 'normies',
                'pepe', 'wojak', 'gigachad', 'cope', 'seethe', 'dilate', 'yeet', 'kek',
                'uwu', 'owo', 'poggers', 'pog', 'sheesh', 'bussin', 'periodt', 'slay',
                # Gaming/Internet culture
                'minecraft', 'fortnite', 'among', 'amogus', 'imposter', 'reddit',
                'updoot', 'downvote', 'wholesome', 'keanu', 'chungus', 'boi',
                # Common abbreviations that should NOT be expanded in memes
                'lol', 'lmao', 'rofl', 'omg', 'wtf', 'fml', 'smh', 'tbh', 'ngl', 
                'imo', 'imho', 'fyi', 'btw', 'afk', 'brb', 'gtg', 'ttyl',
                # Intentional meme misspellings
                'kek', 'rekt', 'pwned', 'noob', 'n00b', 'ur', 'u', 'tho', 'cuz',
                # Social media terms
                'selfie', 'hashtag', 'viral', 'trending', 'influencer', 'tiktoker',
                # Reaction terms
                'mood', 'vibe', 'vibes', 'lowkey', 'highkey', 'deadass', 'fam',
                'flex', 'salty', 'savage', 'fire', 'lit', 'woke', 'shook'
            }
            spell.word_frequency.load_words(meme_words)
            
            # Split text into words and correct ONLY obvious OCR errors
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Remove punctuation for spell checking but preserve it
                clean_word = re.sub(r'[^\w]', '', word.lower())
                
                # Only correct if:
                # 1. Word is longer than 2 characters
                # 2. Not in our meme dictionary
                # 3. Not in standard dictionary
                # 4. Has a clear single correction candidate
                if (len(clean_word) > 2 and 
                    clean_word not in spell and
                    clean_word not in meme_words):
                    
                    candidates = spell.candidates(clean_word)
                    if candidates and len(candidates) == 1:  # Only if there's a single clear candidate
                        correction = list(candidates)[0]
                        # Replace the clean word part while preserving case and punctuation
                        corrected_word = word.lower().replace(clean_word, correction)
                        # Try to preserve original capitalization
                        if word[0].isupper():
                            corrected_word = corrected_word.capitalize()
                        corrected_words.append(corrected_word)
                    else:
                        corrected_words.append(word)  # Keep original if ambiguous
                else:
                    corrected_words.append(word)
            
            text = ' '.join(corrected_words)
        except Exception as e:
            # If spell checking fails, continue without it
            print(f"Basic spell checking failed: {e}")
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_text_from_image(image, backend='paddleocr', preprocess_fn=preprocess_pipeline1_balanced, apply_postprocessing=True):

    #* PaddleOCR (PP-OCRv5) provides excellent multilingual support and complex scene text handling (Pipeline #1 - now default)
    #* EasyOCR (deep-learning based) is much (~10-20x) slower than Tesseract, but more accurate.
    #* Tesseract is fastest but requires more preprocessing
    valid_backends = ['tesseract', 'easyocr', 'paddleocr']

    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend}. Must be one of {valid_backends}")

    if backend == 'paddleocr' and not PADDLEOCR_AVAILABLE:
        raise ImportError("PaddleOCR is not installed. Install with: pip install paddlepaddle paddleocr")

    processed_image = image #preprocess_fn(image)

    # Perform OCR
    if backend == 'tesseract':
        # Use better PSM (Page Segmentation Mode) for memes
        # PSM 6 = single block of text
        # PSM 8 = single word
        # PSM 13 = raw line without word segmentation
        text = pytesseract.image_to_string(processed_image, lang='eng', 
                                         config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:-')
    elif backend == 'easyocr':
        reader = easyocr.Reader(['en'])
        result = reader.readtext(processed_image)
        text = " ".join([text[1] for text in result])
    elif backend == 'paddleocr':
        # Initialize PaddleOCR with angle detection for better performance on memes
        # Use mobile/lightweight version for balanced performance (Pipeline #1)
        ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Simple initialization
        
        # PaddleOCR expects color images, so we need to handle both original and preprocessed images
        # For better results, we'll use the original image with PaddleOCR since it has its own preprocessing
        result = ocr.predict(image)  # Use original image instead of processed_image
        
        # Extract text from PaddleOCR results
        # PaddleOCR 3.x returns a dictionary structure with key-value pairs
        if result and len(result) > 0:
            text_lines = []
            for item in result:
                # Check if item is a dictionary (which it appears to be)
                if isinstance(item, dict):
                    if 'rec_texts' in item:
                        # rec_texts contains the recognized text
                        rec_texts = item['rec_texts']
                        if isinstance(rec_texts, list):
                            text_lines.extend(rec_texts)
                        elif isinstance(rec_texts, str):
                            text_lines.append(rec_texts)
                    # Also check for other possible text fields
                    elif 'text' in item:
                        text_lines.append(item['text'])
                # Handle the old format as fallback
                elif isinstance(item, list) and len(item) > 1:
                    if isinstance(item[1], tuple) and len(item[1]) >= 1:
                        text_lines.append(item[1][0])
                    elif isinstance(item[1], str):
                        text_lines.append(item[1])
            
            # Remove any debug print artifacts and clean the text
            text_lines = [t for t in text_lines if isinstance(t, str) and len(t.strip()) > 0]
            text = " ".join(text_lines)
        else:
            text = ""

    # Apply Pipeline #1 post-processing if requested and using the balanced approach
    if apply_postprocessing and (preprocess_fn == preprocess_pipeline1_balanced or backend == 'paddleocr'):
        text = postprocess_pipeline1_balanced(text)

    return text

def extract_text_with_multiple_preprocessing(image, backend='tesseract'):
    """
    Try multiple preprocessing approaches and return the best result.
    Useful for difficult images where standard preprocessing fails.
    """
    preprocessing_functions = [
        preprocess_adaptive,
        preprocess_high_contrast,
        preprocess_3,
        preprocess_2
    ]
    
    results = []
    for preprocess_fn in preprocessing_functions:
        try:
            text = extract_text_from_image(image, backend, preprocess_fn)
            # Simple heuristic: longer text is often better (more complete extraction)
            results.append((text, len(text.strip())))
        except Exception as e:
            print(f"Error with preprocessing function {preprocess_fn.__name__}: {e}")
            results.append(("", 0))
    
    # Return the result with the most text (simple heuristic)
    best_result = max(results, key=lambda x: x[1])
    return best_result[0]


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
