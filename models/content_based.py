from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_content_based_recommendations(product_id, raw_df, top_n=10, max_features=5000):
    """
    Recommend similar products using TF-IDF on review text.
    
    Algorithm:
    1. Aggregate all review text per product (title + text)
    2. Vectorize product documents using TF-IDF
    3. Compute cosine similarity between product vectors
    4. For the selected product, return top N most similar products
    
    Parameters:
    -----------
    product_id : str
        Product identifier
    raw_df : pd.DataFrame
        Full preprocessed dataframe with reviews.title and reviews.text
    top_n : int
        Number of similar products to return
    max_features : int
        Maximum number of TF-IDF features
    
    Returns:
    --------
    list
        List of similar product_ids (excluding the input product)
    """
    # Aggregate text per product
    product_text = raw_df.groupby('product_id').agg({
        'reviews.title': lambda x: ' '.join(x.fillna('')),
        'reviews.text': lambda x: ' '.join(x.fillna(''))
    }).reset_index()
    
    # Combine title and text
    product_text['content'] = (
        product_text['reviews.title'] + ' ' + product_text['reviews.text']
    ).str.strip()
    
    # Remove empty content
    product_text = product_text[product_text['content'] != '']
    
    if product_text.empty or product_id not in product_text['product_id'].values:
        return []
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = tfidf.fit_transform(product_text['content'])
    except ValueError:
        # Not enough documents or features
        return []
    
    # Find index of the target product
    product_indices = pd.Series(
        product_text.index, 
        index=product_text['product_id']
    ).to_dict()
    
    if product_id not in product_indices:
        return []
    
    idx = product_indices[product_id]
    
    # Compute cosine similarity for this product against all others
    product_vector = tfidf_matrix[idx]
    similarities = cosine_similarity(product_vector, tfidf_matrix).flatten()
    
    # Get indices of most similar products (excluding itself)
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    
    # Convert indices to product IDs
    recommendations = product_text.iloc[similar_indices]['product_id'].tolist()
    
    return recommendations