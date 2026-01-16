from sklearn.decomposition import TruncatedSVD
import numpy as np
from utils.helpers import most_popular_items

def get_svd_recommendations(user_id, R, user_to_idx, idx_to_item, interactions,
                            n_components=20, top_n=10, random_state=42, item_to_idx=None):
    """
    Recommend items using SVD matrix factorization.
    (Minimal fix: safe n_components + optional item_to_idx to remove already-rated items)
    """

    # Check if user exists
    if user_id not in user_to_idx:
        return most_popular_items(R, idx_to_item, top_n)

    user_idx = user_to_idx[user_id]

    n_users, n_items = R.shape

    # IMPORTANT: TruncatedSVD requires n_components < n_items
    max_components = max(1, min(n_users - 1, n_items - 1))
    n_components = int(min(n_components, max_components))

    if n_components < 1:
        return most_popular_items(R, idx_to_item, top_n)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_factors = svd.fit_transform(R)
    item_factors = svd.components_

    user_vector = user_factors[user_idx]
    predicted_scores = np.dot(user_vector, item_factors)

    # Remove already-rated items (works even if item_to_idx not passed)
    predicted_scores = predicted_scores.copy()

    if item_to_idx is not None:
        user_ratings = interactions[interactions['user_id'] == user_id]
        rated_items = set(
            item_to_idx[pid]
            for pid in user_ratings['product_id']
            if pid in item_to_idx
        )
        for item_idx in rated_items:
            predicted_scores[item_idx] = -np.inf
    else:
        # fallback: use the matrix row to mask seen items
        seen = set(R[user_idx].nonzero()[1])
        for item_idx in seen:
            predicted_scores[item_idx] = -np.inf

    top_indices = np.argsort(predicted_scores)[::-1][:top_n]
    top_indices = [i for i in top_indices if predicted_scores[i] > -np.inf]

    return [idx_to_item[i] for i in top_indices]
