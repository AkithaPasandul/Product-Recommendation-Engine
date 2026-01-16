import streamlit as st
import pandas as pd

from data.data_loader import load_and_preprocess_data
from models.popularity import get_popular_products
from models.item_knn import get_item_knn_recommendations
from models.svd_model import get_svd_recommendations
from models.content_based import get_content_based_recommendations
from utils.helpers import build_id_mappings, build_user_item_matrix

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Product Recommendation Engine (Amazon Reviews)",
    layout="wide"
)

st.title("🛍️ Product Recommendation Engine (Amazon Reviews)")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    help="Upload your Amazon reviews CSV file or use default dataset"
)

min_user_reviews = st.sidebar.slider(
    "Min reviews per user",
    1, 20, 3,
    help="Filter out users with fewer reviews"
)

min_product_reviews = st.sidebar.slider(
    "Min reviews per product",
    1, 20, 3,
    help="Filter out products with fewer reviews"
)

top_n = st.sidebar.slider(
    "Top-N recommendations",
    3, 30, 10
)

like_threshold = st.sidebar.slider(
    "Like threshold for KNN",
    1.0, 5.0, 4.0, 0.5
)

min_popularity_reviews = st.sidebar.slider(
    "Min reviews for popularity",
    1, 50, 5
)

# SVD (we'll clamp safely later)
svd_components_requested = st.sidebar.slider(
    "SVD components (requested)",
    5, 100, 20
)

# ---------------------------
# Cache: load data
# ---------------------------
@st.cache_data
def load_data(file, min_user, min_product):
    return load_and_preprocess_data(file, min_user, min_product)

# Default dataset path
file_path = uploaded_file if uploaded_file is not None else "datasets/1429_1.csv"

# ---------------------------
# Helper: build product lookup for nice display
# ---------------------------
@st.cache_data
def build_product_lookup(_raw_df: pd.DataFrame):
    """
    Create lookup: product_id -> {name, brand}
    Dataset has columns: name, brand
    """
    cols = _raw_df.columns.tolist()
    has_name = "name" in cols
    has_brand = "brand" in cols

    # Use first non-null per product_id
    agg_dict = {}
    if has_name:
        agg_dict["name"] = "first"
    if has_brand:
        agg_dict["brand"] = "first"

    if not agg_dict:
        # No metadata available
        return {}

    meta = _raw_df.groupby("product_id", as_index=False).agg(agg_dict)
    lookup = {}
    for r in meta.itertuples(index=False):
        pid = getattr(r, "product_id")
        lookup[pid] = {
            "name": getattr(r, "name", None),
            "brand": getattr(r, "brand", None),
        }
    return lookup

def pretty_recommendations(recommended_ids, product_lookup, title="Recommendations"):
    """
    Convert list of product_ids -> nice table with product name + brand.
    """
    if not recommended_ids:
        st.info("No recommendations available.")
        return

    rows = []
    for i, pid in enumerate(recommended_ids, start=1):
        meta = product_lookup.get(pid, {})
        rows.append({
            "Rank": i,
            "Product Name": meta.get("name") or "(name not available)",
            "Brand": meta.get("brand") or "-",
            "Product ID (ASIN)": pid
        })

    st.subheader(title)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------
# Main
# ---------------------------
try:
    with st.spinner("Loading and preprocessing data..."):
        interactions, raw_df, stats = load_data(file_path, min_user_reviews, min_product_reviews)

    st.sidebar.success("✅ Data loaded successfully!")
    st.sidebar.metric("Rows after cleaning", f"{stats['rows_after_cleaning']:,}")
    st.sidebar.metric("Unique users", f"{stats['unique_users']:,}")
    st.sidebar.metric("Unique products", f"{stats['unique_products']:,}")
    st.sidebar.metric("Total ratings", f"{stats['num_ratings']:,}")

    # Build mappings + matrix
    @st.cache_data
    def get_mappings_and_matrix(_interactions):
        user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_mappings(_interactions)
        R = build_user_item_matrix(_interactions, user_to_idx, item_to_idx)
        return user_to_idx, item_to_idx, idx_to_user, idx_to_item, R

    user_to_idx, item_to_idx, idx_to_user, idx_to_item, R = get_mappings_and_matrix(interactions)

    # Product lookup for display
    product_lookup = build_product_lookup(raw_df)

    # Safe SVD components (avoid ValueError)
    n_users, n_items = R.shape
    max_svd = max(1, min(n_users - 1, n_items - 1))
    svd_components_safe = min(int(svd_components_requested), int(max_svd))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Popularity", "🔗 Item-KNN", "🎯 SVD", "📝 Content-Based"])

    # ---------------------------
    # Tab 1: Popularity
    # ---------------------------
    with tab1:
        st.header("Popularity-Based Recommendations")
        st.write("Shows the most popular products based on weighted rating scores.")

        @st.cache_data
        def get_popularity_table(_interactions, min_reviews, n):
            return get_popular_products(_interactions, min_reviews, n)

        popular_df = get_popularity_table(interactions, min_popularity_reviews, top_n)

        if not popular_df.empty:
            # Add product name/brand columns
            popular_df = popular_df.copy()
            popular_df["Product Name"] = popular_df["product_id"].map(lambda x: product_lookup.get(x, {}).get("name", "(name not available)"))
            popular_df["Brand"] = popular_df["product_id"].map(lambda x: product_lookup.get(x, {}).get("brand", "-"))

            # Reorder columns nicely
            show_cols = ["Product Name", "Brand", "product_id", "mean_rating", "rating_count", "weighted_score"]
            show_cols = [c for c in show_cols if c in popular_df.columns]
            st.dataframe(popular_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.warning("No products meet the minimum review threshold. Try lowering the threshold.")

    # ---------------------------
    # Tab 2: Item-KNN
    # ---------------------------
    with tab2:
        st.header("Item-KNN Collaborative Filtering")
        st.write("Recommends products similar to items the user has liked.")

        all_users = sorted(interactions["user_id"].unique())
        selected_user = st.selectbox("Select a user", options=all_users, key="knn_user")

        if st.button("Get Item-KNN Recommendations", key="knn_btn"):
            with st.spinner("Computing item similarities..."):
                @st.cache_data
                def get_knn_recs(user, _R, _item_to_idx, _idx_to_item, _interactions, threshold, n):
                    return get_item_knn_recommendations(
                        user, _R, _item_to_idx, _idx_to_item, _interactions, threshold, n
                    )

                rec_ids = get_knn_recs(
                    selected_user, R, item_to_idx, idx_to_item, interactions, like_threshold, top_n
                )

            pretty_recommendations(rec_ids, product_lookup, title=f"Top {len(rec_ids)} Recommendations for {selected_user}")

            # Show liked items (by product name)
            liked = interactions[
                (interactions["user_id"] == selected_user) &
                (interactions["rating"] >= like_threshold)
            ]["product_id"].tolist()

            if liked:
                with st.expander(f"📚 Items liked by {selected_user} (rating ≥ {like_threshold})"):
                    for pid in liked[:15]:
                        meta = product_lookup.get(pid, {})
                        st.write(f"• {meta.get('name','(name not available)')}  —  {pid}")

    # ---------------------------
    # Tab 3: SVD
    # ---------------------------
    with tab3:
        st.header("SVD Matrix Factorization")
        st.write("Uses matrix factorization to predict user preferences.")

        all_users = sorted(interactions["user_id"].unique())
        selected_user_svd = st.selectbox("Select a user", options=all_users, key="svd_user")

        st.caption(f"Matrix size: {n_users} users × {n_items} products | Safe max components: {max_svd}")
        st.caption(f"Using SVD components = {svd_components_safe} (auto-clamped to avoid errors)")

        if st.button("Get SVD Recommendations", key="svd_btn"):
            with st.spinner(f"Computing SVD with {svd_components_safe} components..."):
                @st.cache_data
                def get_svd_recs(user, _R, _user_to_idx, _idx_to_item, _interactions, components, n):
                    return get_svd_recommendations(
                        user, _R, _user_to_idx, _idx_to_item, _interactions, components, n
                    )

                rec_ids = get_svd_recs(
                    selected_user_svd, R, user_to_idx, idx_to_item, interactions, svd_components_safe, top_n
                )

            pretty_recommendations(rec_ids, product_lookup, title=f"Top {len(rec_ids)} Recommendations for {selected_user_svd}")

    # ---------------------------
    # Tab 4: Content-based
    # ---------------------------
    with tab4:
        st.header("Content-Based Filtering")
        st.write("Finds similar products based on review text using TF-IDF.")

        all_products = sorted(interactions["product_id"].unique())
        selected_product = st.selectbox("Select a product", options=all_products, key="content_product")

        # show selected product name
        meta = product_lookup.get(selected_product, {})
        st.info(f"Selected: {meta.get('name','(name not available)')}  —  {selected_product}")

        if st.button("Get Similar Products", key="content_btn"):
            with st.spinner("Computing content similarities..."):
                @st.cache_data
                def get_content_recs(product, _raw_df, n):
                    return get_content_based_recommendations(product, _raw_df, n)

                rec_ids = get_content_recs(selected_product, raw_df, top_n)

            pretty_recommendations(rec_ids, product_lookup, title=f"Top {len(rec_ids)} Similar Products")

except FileNotFoundError:
    st.error("❌ Dataset file not found. Please upload a CSV file or check the default path.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.write("Please check your data format and try again.")
