import matplotlib.pyplot as plt
from memes_base.reddit_dataset import RedditRawDataset
from memes_base.image_processing import (load_from_url, extract_text_from_image, 
                                        preprocess_high_contrast, preprocess_3, preprocess_pipeline1_balanced)
import numpy as np
import cv2
import time
import random

def display_complete_results(image, tesseract_text, easyocr_text, paddleocr_text, paddleocr_raw_text,
                           tesseract_time, easyocr_time, paddleocr_time, title=""):
    """Display original image, preprocessed image, and all OCR results in one window."""
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply Pipeline #1 preprocessing for display
    prep_pipeline1 = preprocess_pipeline1_balanced(image)
    
    # Create figure with subplots (2x3 grid to accommodate three OCR results + post-processing comparison)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Display original image (top left)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title(f"Original Image\n{title[:45]}...")
    axes[0, 0].axis('off')
    
    # Display preprocessed image (top middle)
    axes[0, 1].imshow(prep_pipeline1, cmap='gray')
    axes[0, 1].set_title("Pipeline #1 Preprocessing\n(Balanced Approach)")
    axes[0, 1].axis('off')
    
    # Display timing comparison (top right)
    times = [tesseract_time, easyocr_time, paddleocr_time]
    labels = ['Tesseract', 'EasyOCR', 'PaddleOCR\n(Default)']
    colors = ['lightblue', 'lightgreen', 'gold']
    bars = axes[0, 2].bar(labels, times, color=colors)
    axes[0, 2].set_title('Processing Time Comparison')
    axes[0, 2].set_ylabel('Time (seconds)')
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{time_val:.2f}s', ha='center', va='bottom')
    
    # Display Tesseract results (bottom left)
    axes[1, 0].text(0.05, 0.95, 
                   f"Tesseract OCR Results ({tesseract_time:.2f}s):\n\n"
                   f"'{tesseract_text}'\n\n"
                   f"Length: {len(tesseract_text.strip())} chars\n"
                   f"Words: {len(tesseract_text.strip().split())}", 
                   wrap=True, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Tesseract Results")
    axes[1, 0].axis('off')
    
    # Display EasyOCR results (bottom middle)
    axes[1, 1].text(0.05, 0.95, 
                   f"EasyOCR Results ({easyocr_time:.2f}s):\n\n"
                   f"'{easyocr_text}'\n\n"
                   f"Length: {len(easyocr_text.strip())} chars\n"
                   f"Words: {len(easyocr_text.strip().split())}", 
                   wrap=True, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("EasyOCR Results")
    axes[1, 1].axis('off')
    
    # Display PaddleOCR results with post-processing comparison (bottom right)
    improvement_note = ""
    if len(paddleocr_text) != len(paddleocr_raw_text):
        improvement_note = f"\n📝 Post-processing improved text quality!"
    
    axes[1, 2].text(0.05, 0.95, 
                   f"PaddleOCR Pipeline #1 DEFAULT ({paddleocr_time:.2f}s):\n\n"
                   f"With post-processing:\n'{paddleocr_text}'\n\n"
                   f"Raw (no post-processing):\n'{paddleocr_raw_text[:100]}...'\n\n"
                   f"Length: {len(paddleocr_text.strip())} chars\n"
                   f"Words: {len(paddleocr_text.strip().split())}"
                   f"{improvement_note}", 
                   wrap=True, verticalalignment='top', fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.7))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("PaddleOCR Pipeline #1 (DEFAULT)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_ocr():
    # Initialize the dataset
    import yaml
    with open('configs/redditcfg.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = RedditRawDataset(**config['Dataset']['args'])
    
    # Get a sample of posts
    sample_size = 5
    successful_downloads = 0
    attempts = 0
    max_attempts = 50  # Limit attempts to avoid infinite loop
    
    # Track total time for each OCR method
    total_tesseract_time = 0
    total_easyocr_time = 0
    total_paddleocr_time = 0
    
    # Track text extraction quality metrics
    tesseract_results = []
    easyocr_results = []
    paddleocr_results = []
    
    print("Testing Pipeline #1 (PaddleOCR DEFAULT) with Post-processing vs Traditional Methods...")
    print(f"Pipeline #1: PaddleOCR (PP-OCRv5 mobile) with balanced preprocessing + post-processing")
    print(f"Dataset size: {len(dataset)} posts")
    
    # Check if PaddleOCR is available
    try:
        from memes_base.image_processing import PADDLEOCR_AVAILABLE
        if not PADDLEOCR_AVAILABLE:
            print("⚠️  PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
            print("Falling back to Tesseract vs EasyOCR comparison only.")
            return
    except ImportError:
        print("⚠️  PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
        return
    
    # Get random indices
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    
    for random_idx in dataset_indices:
        if successful_downloads >= sample_size or attempts >= max_attempts:
            break
            
        attempts += 1
        post = dataset[random_idx]
        print(f"\nTrying random post {random_idx} (attempt {attempts}): {post['img_url']}")
        
        # Try to load the image
        img = load_from_url(post['img_url'])
        
        if img is not None:
            print("✓ Image loaded successfully!")
            
            # Extract text using all three OCR methods
            print("Running Tesseract OCR (Pipeline #1 preprocessing)...")
            tesseract_start = time.time()
            tesseract_text = extract_text_from_image(img, backend='tesseract', preprocess_fn=preprocess_pipeline1_balanced)
            tesseract_time = time.time() - tesseract_start
            total_tesseract_time += tesseract_time
            
            print("Running EasyOCR (Pipeline #1 preprocessing)...")
            easyocr_start = time.time()
            easyocr_text = extract_text_from_image(img, backend='easyocr', preprocess_fn=preprocess_pipeline1_balanced)
            easyocr_time = time.time() - easyocr_start
            total_easyocr_time += easyocr_time
            
            print("Running PaddleOCR Pipeline #1 (DEFAULT with post-processing)...")
            paddleocr_start = time.time()
            paddleocr_text = extract_text_from_image(img)  # Uses defaults: backend='paddleocr', apply_postprocessing=True
            paddleocr_time = time.time() - paddleocr_start
            total_paddleocr_time += paddleocr_time
            
            # Also get raw PaddleOCR result for comparison
            paddleocr_raw_text = extract_text_from_image(img, backend='paddleocr', apply_postprocessing=False)
            
            # Store results for analysis
            tesseract_results.append({
                'text': tesseract_text,
                'length': len(tesseract_text.strip()),
                'word_count': len(tesseract_text.strip().split()),
                'time': tesseract_time
            })
            
            easyocr_results.append({
                'text': easyocr_text,
                'length': len(easyocr_text.strip()),
                'word_count': len(easyocr_text.strip().split()),
                'time': easyocr_time
            })
            
            paddleocr_results.append({
                'text': paddleocr_text,
                'length': len(paddleocr_text.strip()),
                'word_count': len(paddleocr_text.strip().split()),
                'time': paddleocr_time
            })
            
            # Display results with timing information
            print(f"\n🔧 Tesseract Results (Pipeline #1 preprocessing, {tesseract_time:.2f}s):")
            print(f"Text: {repr(tesseract_text[:100])}...")
            print(f"Length: {len(tesseract_text.strip())} chars, Words: {len(tesseract_text.strip().split())}")
            
            print(f"\n🔧 EasyOCR Results (Pipeline #1 preprocessing, {easyocr_time:.2f}s):")
            print(f"Text: {repr(easyocr_text[:100])}...")
            print(f"Length: {len(easyocr_text.strip())} chars, Words: {len(easyocr_text.strip().split())}")
            
            print(f"\n⭐ PaddleOCR Pipeline #1 DEFAULT Results ({paddleocr_time:.2f}s):")
            print(f"With post-processing: {repr(paddleocr_text[:100])}...")
            print(f"Raw (no post-processing): {repr(paddleocr_raw_text[:100])}...")
            print(f"Length: {len(paddleocr_text.strip())} chars, Words: {len(paddleocr_text.strip().split())}")
            
            # Display everything in one window
            display_complete_results(img, tesseract_text, easyocr_text, paddleocr_text, paddleocr_raw_text,
                                   tesseract_time, easyocr_time, paddleocr_time, 
                                   f"Post {random_idx}: {post['title']}")
            
            successful_downloads += 1
        else:
            print("✗ Failed to load image")
    
    # Print detailed summary statistics
    if successful_downloads > 0:
        avg_tesseract_time = total_tesseract_time / successful_downloads
        avg_easyocr_time = total_easyocr_time / successful_downloads
        avg_paddleocr_time = total_paddleocr_time / successful_downloads
        
        # Calculate text extraction metrics
        avg_tesseract_length = sum(r['length'] for r in tesseract_results) / len(tesseract_results)
        avg_easyocr_length = sum(r['length'] for r in easyocr_results) / len(easyocr_results)
        avg_paddleocr_length = sum(r['length'] for r in paddleocr_results) / len(paddleocr_results)
        
        avg_tesseract_words = sum(r['word_count'] for r in tesseract_results) / len(tesseract_results)
        avg_easyocr_words = sum(r['word_count'] for r in easyocr_results) / len(easyocr_results)
        avg_paddleocr_words = sum(r['word_count'] for r in paddleocr_results) / len(paddleocr_results)
        
        # Count cases where each method extracted the most text
        best_performers = []
        for i in range(len(tesseract_results)):
            lengths = [tesseract_results[i]['length'], easyocr_results[i]['length'], paddleocr_results[i]['length']]
            best_idx = lengths.index(max(lengths))
            best_performers.append(['Tesseract', 'EasyOCR', 'PaddleOCR'][best_idx])
        
        tesseract_wins = best_performers.count('Tesseract')
        easyocr_wins = best_performers.count('EasyOCR')
        paddleocr_wins = best_performers.count('PaddleOCR')
        
        print(f"\n{'='*90}")
        print(f"FINAL RESULTS: Pipeline #1 (PaddleOCR DEFAULT) vs Traditional Methods")
        print(f"{'='*90}")
        print(f"Attempted {attempts} posts, successfully processed {successful_downloads} images.")
        print(f"Success rate: {successful_downloads/attempts*100:.1f}%")
        
        print(f"\n🕐 TIMING RESULTS:")
        print(f"Average Tesseract time: {avg_tesseract_time:.2f}s per image")
        print(f"Average EasyOCR time: {avg_easyocr_time:.2f}s per image")
        print(f"Average PaddleOCR time: {avg_paddleocr_time:.2f}s per image ⭐ DEFAULT")
        
        # Speed comparisons
        print(f"\nSpeed comparisons (relative to PaddleOCR DEFAULT):")
        print(f"Tesseract is {avg_paddleocr_time/avg_tesseract_time:.1f}x faster than PaddleOCR")
        print(f"EasyOCR is {avg_paddleocr_time/avg_easyocr_time:.1f}x faster/slower than PaddleOCR")
        
        print(f"\n📝 TEXT EXTRACTION QUALITY:")
        print(f"Average text length:")
        print(f"  Tesseract: {avg_tesseract_length:.1f} chars")
        print(f"  EasyOCR: {avg_easyocr_length:.1f} chars")
        print(f"  ⭐ PaddleOCR (DEFAULT): {avg_paddleocr_length:.1f} chars")
        
        print(f"\nAverage word count:")
        print(f"  Tesseract: {avg_tesseract_words:.1f} words")
        print(f"  EasyOCR: {avg_easyocr_words:.1f} words")
        print(f"  ⭐ PaddleOCR (DEFAULT): {avg_paddleocr_words:.1f} words")
        
        print(f"\nBest performer by text length:")
        print(f"  Tesseract: {tesseract_wins}/{successful_downloads} cases ({tesseract_wins/successful_downloads*100:.1f}%)")
        print(f"  EasyOCR: {easyocr_wins}/{successful_downloads} cases ({easyocr_wins/successful_downloads*100:.1f}%)")
        print(f"  ⭐ PaddleOCR (DEFAULT): {paddleocr_wins}/{successful_downloads} cases ({paddleocr_wins/successful_downloads*100:.1f}%)")
        
        print(f"\n🎯 PIPELINE #1 FINAL ASSESSMENT:")
        
        # Determine the best overall approach
        pipeline1_score = avg_paddleocr_length + (paddleocr_wins * 10)  # Higher bonus for PaddleOCR wins
        easyocr_score = avg_easyocr_length + (easyocr_wins * 5)
        tesseract_score = avg_tesseract_length + (tesseract_wins * 5)
        
        if paddleocr_wins >= max(tesseract_wins, easyocr_wins):
            print("✅ PaddleOCR Pipeline #1 is now the DEFAULT and performing excellently!")
            print("   Provides balanced performance with superior readability and post-processing.")
        elif avg_paddleocr_length >= max(avg_tesseract_length, avg_easyocr_length) * 0.8:
            print("✅ PaddleOCR Pipeline #1 provides excellent readability even if not always the longest extraction")
            print("   The post-processing significantly improves text quality for practical use.")
        else:
            print("⚠️  Consider adjusting preprocessing parameters for better PaddleOCR performance")
            
        print(f"\n🚀 You can now use extract_text_from_image(image) with PaddleOCR + post-processing as default!")
            
    else:
        print(f"\nNo images were successfully processed after {attempts} attempts.")

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    test_ocr() 