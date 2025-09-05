import streamlit as st
import pdfplumber
import pandas as pd
import easyocr
import os
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fuzzywuzzy import process

# ------------------------ STEP 1: FILE PROCESSING ------------------------

data_dir = "data"
structured_dir = os.path.join(data_dir, "structured")
unstructured_dir = os.path.join(data_dir, "unstructured")

# Initialize EasyOCR Reader
ocr_reader = easyocr.Reader(['en'])

@st.cache_data
def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF."""
    text_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_data.append(text)
    return "\n".join(text_data)

@st.cache_data
def extract_tables_from_pdf(pdf_path):
    """Extract structured tables from a PDF."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                tables.extend(table)
    return tables

@st.cache_data
def extract_data_from_scanned_pdf(pdf_path):
    """Extract text from a scanned PDF using EasyOCR."""
    images = convert_from_path(pdf_path)
    text_data = []
    for img in images:
        result = ocr_reader.readtext(img)
        text_data.extend([text[1] for text in result])
    return " ".join(text_data)

@st.cache_data
def clean_csv_data(csv_path):
    """Read CSV file and clean data."""
    df = pd.read_csv(csv_path)
    expected_columns = ["Cl.", "ExRt", "From", "To", "Valid from", "Exch. Rate", "Ratio (from)", "Ratio (to)"]
    df = df.rename(columns=lambda x: x.strip())
    if all(col in df.columns for col in expected_columns):
        return df[expected_columns]
    else:
        return None

@st.cache_data
def clean_excel_data(excel_path):
    """Read Excel file, clean missing values, and standardize column names."""
    df = pd.read_excel(excel_path)
    df.dropna(how='all', axis=1, inplace=True)
    df.fillna(method='ffill', inplace=True)
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# ------------------------ STEP 2: DATA PROCESSING ------------------------

def fuzzy_match(extracted_text, reference_list):
    """Match extracted text with reference data using fuzzy matching."""
    best_match = process.extractOne(extracted_text, reference_list)
    return best_match[0] if best_match[1] > 80 else None

def process_files():
    """Process all PDFs, CSVs, and Excel files in the structured/unstructured directories."""
    all_data = []
    st.write("Processing structured PDFs, CSVs, and Excel files...")
    
    for file in os.listdir(structured_dir):
        file_path = os.path.join(structured_dir, file)
        tables = None
        
        if file.endswith(".pdf"):
            with st.spinner(f"Extracting data from {file}..."):
                text = extract_text_from_pdf(file_path) or extract_data_from_scanned_pdf(file_path)
                tables = extract_tables_from_pdf(file_path)
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            with st.spinner(f"Cleaning Excel data from {file}..."):
                tables = clean_excel_data(file_path)
        elif file.endswith(".csv"):
            with st.spinner(f"Cleaning CSV data from {file}..."):
                tables = clean_csv_data(file_path)

        if tables is not None:
            all_data.append(pd.DataFrame(tables))
    
    st.success("File processing complete!")
    return pd.concat(all_data, ignore_index=True) if all_data else None

# ------------------------ STEP 3: MACHINE LEARNING ------------------------

def train_model(data):
    """Train a machine learning model to classify extracted data."""
    features = ["ExRt", "Ratio (from)", "Ratio (to)"]
    target = "Exch. Rate"
    
    # Ensure required columns exist
    if not all(col in data.columns for col in features + [target]):
        st.error("Required columns are missing in the dataset!")
        return None

    # Convert columns to numeric, coercing errors
    for col in features + [target]:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert non-numeric to NaN
    st.write(data)
    # Drop rows with NaN values
    #data = data.dropna()
    
    st.write(data)

    # Check if we still have enough data
    if data.empty:
        st.error("Data contains only invalid values after cleanup!")
        return None

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and accuracy
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    st.success(f"🎯 Model Trained! Accuracy: {accuracy:.2f}")
    return model


# ------------------------ STREAMLIT DASHBOARD ------------------------

st.title("📊 Treasury Dashboard for Exchange & Interest Rate Analysis")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Upload & Process Data", "Train Model"])

if menu == "Upload & Process Data":
    st.subheader("📂 Upload & Process Data")
    if st.button("Process Files"):
        processed_data = process_files()
        if processed_data is not None:
            st.success("✅ Data processed successfully!")
            st.dataframe(processed_data.head())
            processed_data.to_csv("clean_financial_data.csv", index=False)
        else:
            st.error("No data extracted. Please check the files.")

elif menu == "Train Model":
    st.subheader("🎯 Train Machine Learning Model")
    if os.path.exists("clean_financial_data.csv"):
        data = pd.read_csv("clean_financial_data.csv")
        model = train_model(data)
        if model:
            st.success("✅ Model trained successfully! Ready for predictions.")
    else:
        st.error("⚠️ No processed data found. Please process data first.")
