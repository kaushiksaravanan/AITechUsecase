import PyPDF2
import re
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for extraction patterns (unchanged)
config = {
    'exchange_rate_patterns': [
        r"([A-Z]{3}/[A-Z]{3})\s*[:,\s]\s*(\d+\.\d+)",
        r"([A-Z]{3}/[A-Z]{3})\s+rate\s+is\s+(\d+\.\d+)"
    ],
    'interest_rate_patterns': [
        r"(\d+[a-zA-Z]+)\s+interest\s+rate\s+is\s+(\d+\.\d+)%?",
        r"(\d+[a-zA-Z]+)\s+rate:\s+(\d+\.\d+)%?"
    ],
    'resource_patterns': [
        r'\b(gold|silver|copper|tea)\b.*\b(price|export|import|sale|volume)\b.*\d+\.\d+',
        r'\b(deed)\b.*\d+\.\d+'
    ],
    'financial_terms': [
        r'\bEBITDA\b.*\d+\.\d+',
        r'\bfree money in circulation\b.*\d+'
    ]
}

def has_selectable_text(text):
    """Check if extracted text is substantial."""
    return len(text.strip()) > 100

def extract_rates_from_text(text, config):
    """Extract exchange and interest rates from text using regex."""
    exchange_rates = {}
    interest_rates = {}
    for pattern in config['exchange_rate_patterns']:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            currency_pair = match.group(1)
            rate = match.group(2)
            exchange_rates[currency_pair] = rate
    for pattern in config['interest_rate_patterns']:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            tenor = match.group(1)
            rate = match.group(2)
            interest_rates[tenor] = rate
    return exchange_rates, interest_rates

def extract_resources_and_terms(text, config):
    """Extract resource and financial term data from text."""
    resource_data = []
    financial_data = []
    for pattern in config['resource_patterns']:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            resource = match.group(1)
            context = match.group(0)
            value = re.search(r'\d+\.\d+', context).group()
            resource_data.append({
                'Resource': resource,
                'Type': context.split(resource)[1].strip().split()[0],
                'Value': float(value)
            })
    for pattern in config['financial_terms']:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            term = match.group(0).split()[0]
            value = re.search(r'\d+\.\d+', match.group(0)).group()
            financial_data.append({
                'Term': term,
                'Value': float(value)
            })
    return pd.DataFrame(resource_data), pd.DataFrame(financial_data)

def clean_rate(rate_str):
    """Clean and convert rate string to float."""
    rate_str = re.sub(r"[^\d\.]", "", rate_str)
    return float(rate_str)

def analyze_market_impact(df_financial, df_resources, df_terms):
    """Analyze impact of resources and terms on financial market."""
    impact_data = []
    if not df_resources.empty and 'Resource' in df_resources.columns:
        gold_data = df_resources[df_resources['Resource'] == 'gold']
        if not gold_data.empty:
            gold_price = gold_data['Value'].mean()
            impact_data.append({
                'Factor': 'Gold Price',
                'Value': gold_price,
                'Impact on Market': 'Potential USD strengthening due to safe-haven demand'
            })
    if not df_terms.empty and 'Term' in df_terms.columns:
        ebitda_data = df_terms[df_terms['Term'] == 'EBITDA']
        if not ebitda_data.empty:
            ebitda_value = ebitda_data['Value'].mean()
            impact_data.append({
                'Factor': 'EBITDA',
                'Value': ebitda_value,
                'Impact on Market': 'Higher EBITDA may indicate stronger corporate health, affecting interest rates'
            })
    return pd.DataFrame(impact_data)

def extract_data_from_pdf(pdf_path, config):
    """Extract all financial data from PDF using PyPDF2."""
    exchange_rates = {}
    interest_rates = {}
    
    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        num_pages = len(pdf.pages)
        
        for page_num in range(num_pages):
            page = pdf.pages[page_num]
            text = page.extract_text()
            
            if has_selectable_text(text):
                # Process text for rates
                text_exchange, text_interest = extract_rates_from_text(text, config)
                for currency, rate in text_exchange.items():
                    if currency not in exchange_rates:
                        exchange_rates[currency] = rate
                for tenor, rate in text_interest.items():
                    if tenor not in interest_rates:
                        interest_rates[tenor] = rate
            else:
                # Perform OCR (note: PyPDF2 doesn't have direct image conversion)
                # This would require additional PDF-to-image conversion library like pdf2image
                logging.info(f"Page {page_num + 1} has no selectable text - OCR needed but not implemented here")
                continue
                
            # Extract resources and financial terms
            df_resources, df_terms = extract_resources_and_terms(text, config)
    
    # Clean rates
    for currency in exchange_rates:
        exchange_rates[currency] = clean_rate(exchange_rates[currency])
    for tenor in interest_rates:
        interest_rates[tenor] = clean_rate(interest_rates[tenor])
    
    # Create financial DataFrame (simplified since no table support)
    df_financial = pd.DataFrame({
        'Currency': list(exchange_rates.keys()),
        'Rate': list(exchange_rates.values())
    })
    df_impact = analyze_market_impact(df_financial, df_resources, df_terms)
    
    return exchange_rates, interest_rates, df_resources, df_terms, df_impact

# Example usage
pdf_path = r'C:\Users\I587436\Documents\New folder\Template\HCLTECH_USECASE\India Outlook FY26_HDFC Bank.pdf'
exchange_rates, interest_rates, df_resources, df_terms, df_impact = extract_data_from_pdf(pdf_path, config)
print("Exchange Rates:", exchange_rates)
print("Interest Rates:", interest_rates)
print("Resource Data:")
print(df_resources)
print("Financial Terms Data:")
print(df_terms)
print("Market Impact Analysis:")
print(df_impact)