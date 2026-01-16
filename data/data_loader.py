import pandas as pd
import streamlit as st

def load_and_preprocess_data(file_path, min_user_reviews=3, min_product_reviews=3):
    
    required_cols = ['reviews.username', 'asins', 'reviews.rating']
    
    # Read CSV
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Drop rows with missing required columns
    df = df.dropna(subset=required_cols)
    
    # Convert rating to numeric
    df['reviews.rating'] = pd.to_numeric(df['reviews.rating'], errors='coerce')
    df = df.dropna(subset=['reviews.rating'])
    
    # Filter valid ratings (1-5)
    df = df[(df['reviews.rating'] >= 1) & (df['reviews.rating'] <= 5)]
    
    # Create user_id from reviews.username (trimmed)
    df['user_id'] = df['reviews.username'].astype(str).str.strip()
    
    # Create product_id from first ASIN in asins column
    def extract_first_asin(asins_str):
        """Extract first ASIN from comma-separated string"""
        if pd.isna(asins_str):
            return None
        asins_str = str(asins_str).strip()
        if ',' in asins_str:
            return asins_str.split(',')[0].strip()
        return asins_str
    
    df['product_id'] = df['asins'].apply(extract_first_asin)
    df = df.dropna(subset=['product_id'])
    
    # Fill missing text fields with empty strings
    if 'reviews.text' in df.columns:
        df['reviews.text'] = df['reviews.text'].fillna('')
    else:
        df['reviews.text'] = ''
    
    if 'reviews.title' in df.columns:
        df['reviews.title'] = df['reviews.title'].fillna('')
    else:
        df['reviews.title'] = ''
    
    # Drop duplicates
    # Keep the most recent review if there are duplicates
    duplicate_cols = ['user_id', 'product_id', 'reviews.rating']
    if 'reviews.text' in df.columns:
        duplicate_cols.append('reviews.text')
    
    df = df.drop_duplicates(subset=duplicate_cols, keep='first')
    
    # Filter sparse users and products
    initial_rows = len(df)
    
    # Iterative filtering to handle cascading effects
    prev_rows = 0
    iterations = 0
    max_iterations = 10
    
    while prev_rows != len(df) and iterations < max_iterations:
        prev_rows = len(df)
        
        # Count reviews per user
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_reviews].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Count reviews per product
        product_counts = df['product_id'].value_counts()
        valid_products = product_counts[product_counts >= min_product_reviews].index
        df = df[df['product_id'].isin(valid_products)]
        
        iterations += 1
    
    # Create interactions dataframe
    interactions = df[['user_id', 'product_id', 'reviews.rating']].copy()
    interactions.columns = ['user_id', 'product_id', 'rating']
    
    # Calculate statistics
    stats = {
        'rows_after_cleaning': len(df),
        'unique_users': df['user_id'].nunique(),
        'unique_products': df['product_id'].nunique(),
        'num_ratings': len(interactions),
        'rows_removed': initial_rows - len(df)
    }
    
    # Warning if too much data was filtered
    if stats['rows_removed'] > initial_rows * 0.8:
        st.warning(
            f"⚠️ {stats['rows_removed']:,} rows ({stats['rows_removed']/initial_rows*100:.1f}%) "
            f"were filtered out. Consider reducing the minimum review thresholds."
        )
    
    return interactions, df, stats