# Product Recommendation Engine (Amazon Reviews)

A complete Streamlit web application for product recommendations using multiple algorithms on Amazon product review data.

## 📁 Project Structure

```
product_recommendation_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── data_loader.py              # Data loading and preprocessing
├── datasets           
├── models/
│   ├── popularity.py              # Popularity-based recommendations
│   ├── item_knn.py                # Item-KNN collaborative filtering
│   ├── svd_model.py               # SVD matrix factorization
│   └── content_based.py           # Content-based filtering (TF-IDF)
└── utils/
    └── helpers.py                  # Shared utility functions
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd product_recommendation_app

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Dataset Requirements

The CSV file must contain these columns:
- **Required**: `reviews.username`, `asins`, `reviews.rating`
- **Optional**: `reviews.text`, `reviews.title`, `reviews.date`

### Data Processing
- **user_id**: Extracted from `reviews.username` (trimmed)
- **product_id**: First ASIN from comma-separated `asins` field
- **rating**: Numeric value between 1-5

## 🎯 Features

### 4 Recommendation Algorithms

#### 1. **Popularity-Based**
- Shows top products based on weighted ratings
- Formula: `score = (v/(v+m)) × R + (m/(v+m)) × C`
- Good for cold-start scenarios and trending products

#### 2. **Item-KNN Collaborative Filtering**
- Recommends items similar to user's liked items
- Uses cosine similarity on item vectors
- Personalized based on user's rating history

#### 3. **SVD Matrix Factorization**
- Latent factor model using TruncatedSVD
- Predicts user preferences for unseen items
- Handles sparse data effectively

#### 4. **Content-Based Filtering**
- Finds similar products based on review text
- Uses TF-IDF vectorization and cosine similarity
- Product-to-product recommendations

## ⚙️ Configuration Options

### Sidebar Controls
- **Min reviews per user** (1-20): Filter sparse users
- **Min reviews per product** (1-20): Filter sparse products
- **Top-N recommendations** (3-30): Number of results to show
- **Like threshold for KNN** (1.0-5.0): Minimum rating to consider as "liked"
- **Min reviews for popularity** (1-50): Threshold for popularity ranking
- **SVD components** (5-100): Number of latent factors

## 🔧 Technical Details

### Data Processing Pipeline
1. Load CSV with `low_memory=False`
2. Drop rows with missing required columns
3. Convert ratings to numeric and filter valid range (1-5)
4. Extract user_id and product_id
5. Fill missing text fields with empty strings
6. Remove duplicates
7. **Iterative filtering** of sparse users/products
8. Build interaction matrix

### Caching Strategy
All expensive operations are cached using `@st.cache_data`:
- Data loading and preprocessing
- ID mappings and sparse matrix construction
- Popularity rankings
- Similarity matrices (item-item, content)
- SVD factorization

### Performance Optimizations
- Sparse matrix operations (CSR format)
- Vectorized computations
- Efficient similarity calculations
- Minimal data reprocessing

## 📝 Usage Examples

### Upload Your Data
1. Click "Upload CSV Dataset" in sidebar
2. Or place your CSV at `/mnt/data/1429_1.csv` as default

### Get Recommendations

**Popularity**: View top products instantly

**Item-KNN**: 
1. Select a user from dropdown
2. Click "Get Item-KNN Recommendations"
3. View personalized recommendations

**SVD**: 
1. Select a user
2. Adjust SVD components if needed
3. Click "Get SVD Recommendations"

**Content-Based**: 
1. Select a product
2. Click "Get Similar Products"
3. See products with similar review content

## 🛠️ Customization

### Modify Algorithms
Each model is in its own file under `models/`:
- Edit algorithm parameters
- Change similarity metrics
- Adjust scoring formulas

### Add New Models
1. Create new file in `models/`
2. Implement recommendation function
3. Add new tab in `app.py`
4. Cache results with `@st.cache_data`

### Adjust UI
Modify `app.py` to:
- Add new sidebar controls
- Change tab layouts
- Customize metrics display
- Add data visualizations

## ⚠️ Troubleshooting

### "Dataset file not found"
- Upload a CSV file via sidebar, or
- Place your CSV at the default path `/mnt/data/1429_1.csv`

### "Too much data filtered out"
- Reduce "Min reviews per user/product" sliders
- Check data quality and completeness

### "No recommendations available"
- User may not have enough ratings
- Try different users or adjust thresholds
- Check if user exists in filtered dataset

### Performance Issues
- Reduce dataset size via filtering
- Lower SVD components
- Reduce max_features for content-based filtering

## 📦 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scipy**: Sparse matrices
- **scikit-learn**: ML algorithms (TF-IDF, SVD, cosine similarity)

## 🎨 Code Quality

### Design Principles
- **Separation of Concerns**: UI, data, models, utils in separate modules
- **DRY**: Shared utilities in `helpers.py`
- **Caching**: Minimize recomputation
- **Error Handling**: Graceful failures with user feedback
- **Documentation**: Docstrings for all functions

### Best Practices
- Type hints in docstrings
- Clear variable names
- Modular functions (single responsibility)
- Comprehensive comments
- Professional UI/UX

## 📄 License

This is an educational project for demonstration purposes.

## 🤝 Contributing

Feel free to extend this project with:
- Additional recommendation algorithms
- Data visualizations
- A/B testing framework
- Model evaluation metrics
- Export functionality
- User feedback collection

---
