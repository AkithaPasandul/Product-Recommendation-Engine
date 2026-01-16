import pandas as pd

def get_popular_products(interactions, min_reviews=5, top_n=10):
    rating_col = "rating" if "rating" in interactions.columns else "reviews.rating"

    C = interactions[rating_col].mean()

    product_stats = interactions.groupby("product_id").agg(
        mean_rating=(rating_col, "mean"),
        rating_count=(rating_col, "count")
    ).reset_index()

    product_stats = product_stats[product_stats["rating_count"] >= min_reviews]
    if product_stats.empty:
        return pd.DataFrame()

    m = min_reviews
    product_stats["weighted_score"] = (
        (product_stats["rating_count"] / (product_stats["rating_count"] + m)) * product_stats["mean_rating"] +
        (m / (product_stats["rating_count"] + m)) * C
    )

    top_products = product_stats.nlargest(top_n, "weighted_score")
    top_products["mean_rating"] = top_products["mean_rating"].round(2)
    top_products["weighted_score"] = top_products["weighted_score"].round(3)

    return top_products[["product_id", "mean_rating", "rating_count", "weighted_score"]]
