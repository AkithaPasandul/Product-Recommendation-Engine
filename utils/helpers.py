import pandas as pd
from scipy.sparse import csr_matrix

def build_id_mappings(interactions):
    """
    Build bidirectional mappings between user/item IDs and matrix indices.
    
    Parameters:
    -----------
    interactions : pd.DataFrame
        DataFrame with columns [user_id, product_id, rating]
    
    Returns:
    --------
    user_to_idx : dict
        Mapping from user_id to matrix row index
    item_to_idx : dict
        Mapping from product_id to matrix column index
    idx_to_user : dict
        Mapping from matrix row index to user_id
    idx_to_item : dict
        Mapping from matrix column index to product_id
    """
    unique_users = sorted(interactions['user_id'].unique())
    unique_items = sorted(interactions['product_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item


def build_user_item_matrix(interactions, user_to_idx, item_to_idx):
    """
    Build a sparse user-item rating matrix.
    
    Parameters:
    -----------
    interactions : pd.DataFrame
        DataFrame with columns [user_id, product_id, rating]
    user_to_idx : dict
        Mapping from user_id to matrix row index
    item_to_idx : dict
        Mapping from product_id to matrix column index
    
    Returns:
    --------
    R : scipy.sparse.csr_matrix
        Sparse user-item rating matrix of shape (n_users, n_items)
    """
    # Map IDs to indices
    row_indices = interactions['user_id'].map(user_to_idx).values
    col_indices = interactions['product_id'].map(item_to_idx).values
    ratings = interactions['rating'].values
    
    # Create sparse matrix
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    
    R = csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(n_users, n_items)
    )
    
    return R


def most_popular_items(R, idx_to_item, top_n=10):
    """
    Get the most popular items based on rating counts.
    
    Parameters:
    -----------
    R : scipy.sparse.csr_matrix
        User-item rating matrix
    idx_to_item : dict
        Mapping from matrix column index to product_id
    top_n : int
        Number of top items to return
    
    Returns:
    --------
    list
        List of product_ids sorted by popularity
    """
    # Count non-zero ratings per item
    item_counts = (R > 0).sum(axis=0).A1  # Convert to 1D array
    
    # Get top N item indices
    top_indices = item_counts.argsort()[::-1][:top_n]
    
    # Convert indices to product IDs
    popular_items = [idx_to_item[idx] for idx in top_indices]
    
    return popular_items


def filter_seen_items(candidate_scores, user_ratings):
    """
    Filter out items that the user has already rated.
    
    Parameters:
    -----------
    candidate_scores : dict
        Dictionary mapping item_idx to score
    user_ratings : scipy.sparse matrix row
        Sparse row vector of user's ratings
    
    Returns:
    --------
    dict
        Filtered candidate_scores with seen items removed
    """
    # Get indices of items the user has rated
    seen_items = set(user_ratings.nonzero()[1])
    
    # Filter out seen items
    filtered_scores = {
        item_idx: score 
        for item_idx, score in candidate_scores.items() 
        if item_idx not in seen_items
    }
    
    return filtered_scores


def get_user_ratings(user_id, R, user_to_idx):
    """
    Get a user's rating vector from the sparse matrix.
    
    Parameters:
    -----------
    user_id : str
        User identifier
    R : scipy.sparse.csr_matrix
        User-item rating matrix
    user_to_idx : dict
        Mapping from user_id to matrix row index
    
    Returns:
    --------
    scipy.sparse matrix row or None
        User's rating vector, or None if user not found
    """
    if user_id not in user_to_idx:
        return None
    
    user_idx = user_to_idx[user_id]
    return R[user_idx]


def get_item_ratings(product_id, R, item_to_idx):
    """
    Get an item's rating vector from the sparse matrix.
    
    Parameters:
    -----------
    product_id : str
        Product identifier
    R : scipy.sparse.csr_matrix
        User-item rating matrix
    item_to_idx : dict
        Mapping from product_id to matrix column index
    
    Returns:
    --------
    scipy.sparse matrix column or None
        Item's rating vector, or None if item not found
    """
    if product_id not in item_to_idx:
        return None
    
    item_idx = item_to_idx[product_id]
    return R[:, item_idx]