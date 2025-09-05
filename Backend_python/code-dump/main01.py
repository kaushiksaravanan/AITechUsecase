import fitz  # PyMuPDF
import easyocr
import pandas as pd
import re
from transformers import pipeline
import logging
import io
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tqdm import tqdm

# Set up logging (reduced verbosity)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize EasyOCR with GPU support if available
reader = easyocr.Reader(['en'], gpu=True)

# Use a finance-specific sentiment model
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    truncation=True,
    max_length=512
)

# Precompile regex patterns for better performance
CURRENCY_PATTERN = re.compile(r'\b[A-Z]{3}/[A-Z]{3}\b|\b[A-Z]{3}\s+[A-Z]{3}\b')
RATE_PATTERN = re.compile(r'\d+\.\d{2,6}|\d+\s*\.\s*\d{2,6}')
DATE_PATTERN = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b')
PERCENT_PATTERN = re.compile(r'\d+\.?\d*%\b')


@lru_cache(maxsize=128)
def extract_with_regex(text, pattern_type):
    """Extract matches from text using precompiled regex patterns with caching."""
    if pattern_type == 'currency':
        return list(set(CURRENCY_PATTERN.findall(text)))
    elif pattern_type == 'rate':
        return list(set(RATE_PATTERN.findall(text)))
    elif pattern_type == 'date':
        return list(set(DATE_PATTERN.findall(text)))
    elif pattern_type == 'percent':
        return list(set(PERCENT_PATTERN.findall(text)))
    return []

def extract_text_from_pdf(pdf_path):
    """Extract raw text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def extract_data_from_pdf(pdf_path):
    """Extract data from a PDF file with optimized processing and save extracted text."""
    data = []
    doc = fitz.open(pdf_path)
    output_text_folder = os.path.join(os.path.dirname(pdf_path), "extracted_text")
    os.makedirs(output_text_folder, exist_ok=True)
    
    # Save extracted text to file
    text_file_path = os.path.join(output_text_folder, f"{os.path.basename(pdf_path)}_extracted.txt")
    full_text = extract_text_from_pdf(pdf_path)
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    # Progress bar for pages within each PDF
    for page_num, page in enumerate(tqdm(doc, desc=f"Processing pages in {os.path.basename(pdf_path)}", leave=False), 1):
        # Optimize pixmap generation with lower DPI for speed
        pix = page.get_pixmap(dpi=150)
        img_byte_arr = io.BytesIO(pix.tobytes("png"))
        
        # Perform OCR with minimal post-processing
        ocr_result = reader.readtext(
            img_byte_arr.getvalue(),
            detail=0,  # Only get text, skip coordinates
            paragraph=True  # Faster text grouping
        )
        ocr_text = ' '.join(ocr_result)
        
        # Extract various types of financial data
        currencies = extract_with_regex(ocr_text, 'currency')
        rates = extract_with_regex(ocr_text, 'rate')
        dates = extract_with_regex(ocr_text, 'date')
        percentages = extract_with_regex(ocr_text, 'percent')
        
        # Quick sentiment analysis on finance-relevant text
        sentiment = sentiment_analyzer(ocr_text)[0]
        
        # Efficient data combination
        min_length = min(len(currencies), len(rates))
        data.extend([{
            'Currency': currencies[i] if i < len(currencies) else None,
            'Exch. Rate': rates[i] if i < len(rates) else None,
            'Date': dates[i] if i < len(dates) else None,
            'Percentage': percentages[i] if i < len(percentages) else None,
            'Sentiment': 'POSITIVE' if sentiment['label'] == 'positive' else 
                        'NEGATIVE' if sentiment['label'] == 'negative' else 'NEUTRAL',
            'Sentiment Score': sentiment['score'],
            'Page Number': page_num
        } for i in range(max(1, min_length))])  # Ensure at least one row per page
    
    doc.close()
    return pd.DataFrame(data)

def process_pdf(pdf_path):
    """Wrapper for parallel processing."""
    try:
        df = extract_data_from_pdf(pdf_path)
        df['source_file'] = os.path.basename(pdf_path)
        return df
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        return pd.DataFrame()

# Folder containing PDFs
pdf_folder = r'C:\Users\I587436\Documents\New folder\Template\HCLTECH_USECASE'
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

# Parallel processing of PDFs with tqdm
all_data = []
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    # Progress bar for PDFs
    all_data = list(tqdm(
        executor.map(process_pdf, pdf_files),
        total=len(pdf_files),
        desc="Processing PDF files"
    ))

# Combine and save results
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    output_csv_path = os.path.join(pdf_folder, "processed_data_with_sentiment.csv")
    final_df.to_csv(output_csv_path, index=False)
    
    print(f"Data saved to: {output_csv_path}")
    print(f"Extracted text files saved in: {os.path.join(pdf_folder, 'extracted_text')}")
    print("\nFirst few rows of processed data:")
    print(final_df.head())
else:
    print("No data extracted from PDFs.")