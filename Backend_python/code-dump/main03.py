import PyPDF2
from pdf2image import convert_from_path
import easyocr
import tabula
import re
from fuzzywuzzy import fuzz
from textblob import TextBlob
import pandas as pd
from dateutil import parser
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Define all ISO 4217 currency codes
currency_codes = [
    'AED', 'AFN', 'ALL', 'AMD', 'ANG', 'AOA', 'ARS', 'AUD', 'AWG', 'AZN', 'BAM', 'BBD', 'BDT', 'BGN', 'BHD', 'BIF',
    'BMD', 'BND', 'BOB', 'BRL', 'BSD', 'BTN', 'BWP', 'BYN', 'BZD', 'CAD', 'CDF', 'CHF', 'CLP', 'CNY', 'COP', 'CRC',
    'CUC', 'CUP', 'CVE', 'CZK', 'DJF', 'DKK', 'DOP', 'DZD', 'EGP', 'ERN', 'ETB', 'EUR', 'FJD', 'FKP', 'GBP', 'GEL',
    'GGP', 'GHS', 'GIP', 'GMD', 'GNF', 'GTQ', 'GYD', 'HKD', 'HNL', 'HRK', 'HTG', 'HUF', 'IDR', 'ILS', 'IMP', 'INR',
    'IQD', 'IRR', 'ISK', 'JEP', 'JMD', 'JOD', 'JPY', 'KES', 'KGS', 'KHR', 'KMF', 'KPW', 'KRW', 'KWD', 'KYD', 'KZT',
    'LAK', 'LBP', 'LKR', 'LRD', 'LSL', 'LYD', 'MAD', 'MDL', 'MGA', 'MKD', 'MMK', 'MNT', 'MOP', 'MRU', 'MUR', 'MVR',
    'MWK', 'MXN', 'MYR', 'MZN', 'NAD', 'NGN', 'NIO', 'NOK', 'NPR', 'NZD', 'OMR', 'PAB', 'PEN', 'PGK', 'PHP', 'PKR',
    'PLN', 'PYG', 'QAR', 'RON', 'RSD', 'RUB', 'RWF', 'SAR', 'SBD', 'SCR', 'SDG', 'SEK', 'SGD', 'SHP', 'SLL', 'SOS',
    'SRD', 'SSP', 'STN', 'SVC', 'SYP', 'SZL', 'THB', 'TJS', 'TMT', 'TND', 'TOP', 'TRY', 'TTD', 'TWD', 'TZS', 'UAH',
    'UGX', 'USD', 'UYU', 'UZS', 'VES', 'VND', 'VUV', 'WST', 'XAF', 'XCD', 'XDR', 'XOF', 'XPF', 'YER', 'ZAR', 'ZMW',
    'ZWL'
]

# Define specific data points with patterns and units
specific_data_points = {
    'Gold Price': {
        'patterns': [r'gold price:\s*(\d+\.\d+)', r'gold is trading at\s*(\d+\.\d+)'],
        'unit': 'USD/oz'
    },
    'Silver Price': {
        'patterns': [r'silver price:\s*(\d+\.\d+)', r'silver is at\s*(\d+\.\d+)'],
        'unit': 'USD/oz'
    },
    'Copper Price': {
        'patterns': [r'copper price:\s*(\d+\.\d+)', r'copper is at\s*(\d+\.\d+)'],
        'unit': 'USD/lb'
    },
    'Tea Export Volume': {
        'patterns': [r'tea export volume:\s*(\d+)', r'exported\s*(\d+)\s*tons of tea'],
        'unit': 'tons'
    },
    'Tea Import Volume': {
        'patterns': [r'tea import volume:\s*(\d+)', r'imported\s*(\d+)\s*tons of tea'],
        'unit': 'tons'
    },
    'Tea Price': {
        'patterns': [r'tea price:\s*(\d+\.\d+)', r'tea sells for\s*(\d+\.\d+)'],
        'unit': 'USD/kg'
    },
    'EBIDA Company X': {
        'patterns': [r'Company X EBIDA:\s*(\d+\.\d+)\s*billion', r'EBIDA\s*(\d+\.\d+)'],
        'unit': 'billion USD'
    },
    'Money Supply M2': {
        'patterns': [r'money supply m2:\s*(\d+\.\d+)\s*trillion'],
        'unit': 'trillion USD'
    },
    'Currency in Circulation': {
        'patterns': [r'currency in circulation:\s*(\d+\.\d+)\s*billion'],
        'unit': 'billion USD'
    }
}

# Additional table headers for exchange rates from CSV and PDF
table_headers = [
    'From', 'To', 'Date', 'Exch. Rate', 'Currency', 'Interest Rate',
    'Gold Price', 'Silver Price', 'Copper Price', 'Tea Export Volume', 'Tea Import Volume', 'Tea Price',
    'EBIDA Company X', 'Money Supply M2', 'Currency in Circulation',
    'Sentiment Polarity', 'Sentiment Subjectivity'
]

# Load embedding model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader for English

def is_usable_text(text):
    """
    Check if the extracted text is usable based on length and financial keywords.
    
    Args:
        text (str): Extracted text from PDF.
    
    Returns:
        bool: True if text is usable, False otherwise.
    """
    financial_keywords = [
        'bank', 'rate', 'exchange', 'interest', 'currency', 'financial', 'report', 'statement',
        'gold', 'silver', 'copper', 'tea', 'export', 'import', 'sale', 'price', 'production',
        'EBIDA', 'money supply', 'currency in circulation'
    ]
    return len(text) > 100 and any(keyword in text.lower() for keyword in financial_keywords)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2 for searchable PDFs or EasyOCR for scanned PDFs.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text.
    """
    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_file.close()
        if is_usable_text(text):
            return text
        else:
            raise Exception("Text not usable, proceeding to OCR")
    except Exception:
        # Convert PDF to images for OCR
        images = convert_from_path(pdf_path)
        ocr_text = ''
        for image in images:
            result = reader.readtext(image)
            ocr_text += ' '.join([res[1] for res in result]) + '\n'
        return ocr_text

def extract_tables_from_pdf(pdf_path):
    """
    Extract tables from a PDF file using tabula-py.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        list: List of pandas DataFrames representing tables.
    """
    try:
        tables = tabula.read_pdf(pdf_path, pages='all')
        return tables
    except Exception as e:
        print(f"Table extraction failed: {e}")
        return []

def get_embedding_similarity(sentence1, sentence2):
    """
    Calculate semantic similarity between two sentences using embeddings.
    
    Args:
        sentence1 (str): First sentence.
        sentence2 (str): Second sentence.
    
    Returns:
        float: Cosine similarity score.
    """
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

def fuzzy_match(term, terms_list, threshold=80):
    """
    Perform fuzzy matching to find similar terms.
    
    Args:
        term (str): Term to match.
        terms_list (list): List of terms to compare against.
        threshold (int): Similarity threshold (default: 80).
    
    Returns:
        str: Matched term or None if no match.
    """
    for t in terms_list:
        if fuzz.ratio(term.lower(), t.lower()) > threshold:
            return t
    return None

def extract_rates_from_text(text, currency_codes):
    """
    Extract exchange and interest rates from text using regex and embeddings.
    
    Args:
        text (str): Extracted text from PDF.
        currency_codes (list): List of valid currency codes.
    
    Returns:
        tuple: Lists of exchange rates and interest rates.
    """
    exchange_rates = []
    interest_rates = []
    
    # Direct patterns for exchange rates
    exchange_patterns = [
        r'(\w{3})\s*/\s*(\w{3})\s*:\s*(\d+\.\d+)',
        r'(\w{3})\s+to\s+(\w{3})\s+:\s+(\d+\.\d+)',
        r'exchange rate for\s+(\w{3})\s+against\s+(\w{3})\s+is\s+(\d+\.\d+)'
    ]
    
    # Direct patterns for interest rates
    interest_patterns = [
        r'(\w{3})\s+interest rate\s+:\s+(\d+\.\d+)%',
        r'interest rate in\s+(\w{3})\s+is\s+(\d+\.\d+)%'
    ]
    
    # Extract direct exchange rates
    for pattern in exchange_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            from_curr, to_curr, rate = match
            if from_curr in currency_codes and to_curr in currency_codes:
                exchange_rates.append((from_curr, to_curr, float(rate)))
    
    # Extract direct interest rates
    for pattern in interest_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            curr, rate = match
            if curr in currency_codes:
                interest_rates.append((curr, float(rate)))
    
    # Handle indirect mentions using embeddings
    sentences = text.split('.')
    for sentence in sentences:
        words = sentence.split()
        currencies_in_sentence = [word for word in words if word in currency_codes]
        numbers_in_sentence = [word for word in words if re.match(r'\d+\.\d+', word)]
        percentages_in_sentence = [word for word in words if '%' in word]
        
        # Check for potential exchange rate (2 currencies and a number)
        if len(currencies_in_sentence) >= 2 and numbers_in_sentence:
            from_curr, to_curr = currencies_in_sentence[:2]
            rate = numbers_in_sentence[0]
            example_sentence = f"The exchange rate for {from_curr} to {to_curr} is {rate}"
            similarity = get_embedding_similarity(sentence, example_sentence)
            if similarity > 0.7:
                exchange_rates.append((from_curr, to_curr, float(rate)))
        
        # Check for potential interest rate (1 currency, rate/interest keywords, and a percentage)
        elif len(currencies_in_sentence) == 1 and (percentages_in_sentence or ('rate' in words or 'interest' in words)):
            curr = currencies_in_sentence[0]
            rate = percentages_in_sentence[0].replace('%', '') if percentages_in_sentence else numbers_in_sentence[0]
            example_sentence = f"The interest rate in {curr} is {rate}%"
            similarity = get_embedding_similarity(sentence, example_sentence)
            if similarity > 0.7:
                interest_rates.append((curr, float(rate)))
    
    return exchange_rates, interest_rates

def extract_specific_data_from_text(text, specific_data_points):
    """
    Extract specific financial data points (e.g., gold price, tea exports) from text.
    
    Args:
        text (str): Extracted text from PDF.
        specific_data_points (dict): Dictionary of data points with patterns and units.
    
    Returns:
        list: Extracted specific data points.
    """
    extracted_data = []
    for data_point, config in specific_data_points.items():
        for pattern in config['patterns']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match
                extracted_data.append((data_point, float(value), config['unit'], extract_date_from_text(text)))
    
    # Embedding-based search for indirect mentions
    sentences = text.split('.')
    for sentence in sentences:
        for data_point, config in specific_data_points.items():
            description = f"{data_point} is"
            similarity = get_embedding_similarity(sentence, description)
            if similarity > 0.7:
                numbers = re.findall(r'\d+\.\d+|\d+', sentence)
                if numbers:
                    value = numbers[0]
                    extracted_data.append((data_point, float(value), config['unit'], extract_date_from_text(text)))
    return extracted_data

def extract_date_from_text(text):
    """
    Extract the first valid date from the text.
    
    Args:
        text (str): Extracted text from PDF.
    
    Returns:
        date: Extracted date object or None.
    """
    date_patterns = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+\s+\d{1,2},\s+\d{4}', text)
    for pattern in date_patterns:
        try:
            date = parser.parse(pattern).date()
            return date
        except:
            pass
    return None

def get_sentiment(text):
    """
    Perform sentiment analysis on the text using TextBlob.
    
    Args:
        text (str): Extracted text from PDF.
    
    Returns:
        tuple: Sentiment polarity and subjectivity.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def process_pdf(pdf_path, currency_codes, specific_data_points):
    """
    Process a PDF file to extract exchange rates, interest rates, and specific financial data.
    
    Args:
        pdf_path (str): Path to the PDF file.
        currency_codes (list): List of valid currency codes.
        specific_data_points (dict): Dictionary of specific data points with patterns and units.
    
    Returns:
        tuple: DataFrames for exchange rates, interest rates, and specific data.
    """
    # Extract text and tables
    text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_pdf(pdf_path)
    
    # Extract rates from text
    exchange_rates, interest_rates = extract_rates_from_text(text, currency_codes)
    
    # Extract specific data from text
    specific_data = extract_specific_data_from_text(text, specific_data_points)
    
    # Extract rates and specific data from tables
    for table in tables:
        if isinstance(table, pd.DataFrame):
            # Exchange rates
            exchange_cols = ['From', 'To', 'Rate', 'Exchange Rate', 'Exch. Rate', 'Currency Pair']
            interest_cols = ['Currency', 'Interest Rate', 'Rate (%)']
            specific_cols = list(specific_data_points.keys())
            
            for index, row in table.iterrows():
                # Exchange rates
                for from_col in exchange_cols:
                    for to_col in exchange_cols:
                        for rate_col in exchange_cols:
                            if from_col in table.columns and to_col in table.columns and rate_col in table.columns:
                                from_curr = row[from_col]
                                to_curr = row[to_col]
                                rate = row[rate_col]
                                if isinstance(from_curr, str) and isinstance(to_curr, str) and isinstance(rate, (int, float, str)):
                                    if from_curr in currency_codes and to_curr in currency_codes:
                                        exchange_rates.append((from_curr, to_curr, float(rate)))
                
                # Interest rates
                for curr_col in interest_cols:
                    for rate_col in interest_cols:
                        if curr_col in table.columns and rate_col in table.columns:
                            curr = row[curr_col]
                            rate = row[rate_col]
                            if isinstance(curr, str) and isinstance(rate, (int, float, str)):
                                if curr in currency_codes:
                                    interest_rates.append((curr, float(rate)))
                
                # Specific data
                for col in specific_cols:
                    if col in table.columns:
                        value = row[col]
                        if isinstance(value, (int, float, str)):
                            unit = specific_data_points[col]['unit']
                            specific_data.append((col, float(value), unit, extract_date_from_text(text)))
    
    # Extract date and sentiment
    date = extract_date_from_text(text)
    polarity, subjectivity = get_sentiment(text)
    
    # Structure exchange rate data
    exchange_df = pd.DataFrame(exchange_rates, columns=['From', 'To', 'Exch. Rate'])
    exchange_df['Date'] = date
    exchange_df['Sentiment Polarity'] = polarity
    exchange_df['Sentiment Subjectivity'] = subjectivity
    
    # Structure interest rate data
    interest_df = pd.DataFrame(interest_rates, columns=['Currency', 'Interest Rate'])
    interest_df['Date'] = date
    interest_df['Sentiment Polarity'] = polarity
    interest_df['Sentiment Subjectivity'] = subjectivity
    
    # Structure specific data
    specific_df = pd.DataFrame(specific_data, columns=['Data Point', 'Value', 'Unit', 'Date'])
    specific_df['Sentiment Polarity'] = polarity
    specific_df['Sentiment Subjectivity'] = subjectivity
    
    return exchange_df, interest_df, specific_df

def combine_with_csv(exchange_df, csv_path):
    """
    Combine extracted exchange rate data with CSV data.
    
    Args:
        exchange_df (DataFrame): Exchange rate data from PDF.
        csv_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: Combined exchange rate data.
    """
    csv_df = pd.read_csv(csv_path)
    csv_df['Date'] = pd.to_datetime(csv_df['Valid from'], format='%d.%m.%Y')
    csv_df = csv_df[['From', 'To', 'Date', 'Exch. Rate']]
    combined_df = pd.concat([exchange_df, csv_df], ignore_index=True)
    return combined_df

def prepare_for_time_series(df):
    """
    Prepare DataFrame for time series analysis by setting 'Date' as index.
    
    Args:
        df (DataFrame): Combined data.
    
    Returns:
        DataFrame: Prepared time series data.
    """
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Example usage
if __name__ == "__main__":
    pdf_path =  b'C:\Users\I587436\Documents\New folder\Template\HCLTECH_USECASE\India Outlook FY26_HDFC Bank.pdf'  # Replace with your PDF path
    csv_path = b'C:\Users\I587436\Documents\New folder\Template\HCLTECH_USECASE\Exchange Rate_Bloomberg_TCURR_ 06 Feb 22.xlsx'  # Replace with your CSV path

    # Process PDF to extract data
    exchange_df, interest_df, specific_df = process_pdf(pdf_path, currency_codes, specific_data_points)
    
    # Combine exchange rate data with CSV
    combined_exchange_df = combine_with_csv(exchange_df, csv_path)
    
    # Prepare for time series analysis
    ts_exchange_df = prepare_for_time_series(combined_exchange_df)
    ts_specific_df = prepare_for_time_series(specific_df)
    
    # Print extracted data
    print("Exchange Rates:")
    print(ts_exchange_df.head())
    print("\nInterest Rates:")
    print(interest_df.head())
    print("\nSpecific Financial Data:")
    print(ts_specific_df.head())
    
    # Visualize exchange rates
    plt.figure(figsize=(10, 6))
    ts_exchange_df['Exch. Rate'].plot()
    plt.title('Time Series of Exchange Rates')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.show()
    
    # Visualize specific data (e.g., Gold Price)
    if not ts_specific_df.empty and 'Gold Price' in ts_specific_df['Data Point'].values:
        gold_df = ts_specific_df[ts_specific_df['Data Point'] == 'Gold Price']
        plt.figure(figsize=(10, 6))
        gold_df['Value'].plot()
        plt.title('Time Series of Gold Price')
        plt.xlabel('Date')
        plt.ylabel('Gold Price (USD/oz)')
        plt.show()