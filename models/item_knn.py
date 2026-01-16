import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import most_popular_items

def get_item_knn_recommendations(user_id, R, item_to_idx, idx_to_item, interactions,
                                like_threshold=4.0, top_n=10):
    """
    Recommend items using item-item collaborative filtering with cosine similarity.
    """

    # rating column compatibility
    rating_col = "rating" if "rating" in interactions.columns else "reviews.rating"

    user_ratings = interactions[interactions['user_id'] == user_id]

    if user_ratings.empty:
        return most_popular_items(R, idx_to_item, top_n)

    liked_items = user_ratings[user_ratings[rating_col] >= like_threshold]

    if liked_items.empty:
        return most_popular_items(R, idx_to_item, top_n)

    liked_item_data = []
    for _, row in liked_items.iterrows():
        pid = row["product_id"]
        if pid in item_to_idx:
            liked_item_data.append((item_to_idx[pid], float(row[rating_col])))

    if not liked_item_data:
        return most_popular_items(R, idx_to_item, top_n)

    # item-item similarity (compute once)
    item_similarity = cosine_similarity(R.T, dense_output=False)

    candidate_scores = {}

    for candidate_idx in range(R.shape[1]):
        score = 0.0
        for liked_idx, user_rating in liked_item_data:
            sim_val = item_similarity[candidate_idx, liked_idx]
            sim_val = float(sim_val) if np.isscalar(sim_val) else float(sim_val.toarray()[0][0])
            score += sim_val * float(user_rating)
        candidate_scores[candidate_idx] = score

    rated_items = set(
        item_to_idx[pid] for pid in user_ratings["product_id"] if pid in item_to_idx
    )
    candidate_scores = {i: s for i, s in candidate_scores.items() if i not in rated_items}

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [idx_to_item[i] for i, _ in sorted_candidates]
