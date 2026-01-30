# Customer Churn Analysis Dashboard

A production-ready Streamlit application for analyzing customer churn patterns and making predictions using a Random Forest classifier.

## App Structure

This dashboard provides:
1. **KPI Cards** - Total customers, churn rate, retention rate, and average tenure
2. **Interactive Filters** - Filter by contract type, tenure range, and monthly charges
3. **Churn Insights** - Visual analysis of churn patterns by tenure, contract type, and charges
4. **Model Performance** - Accuracy, precision, recall, ROC-AUC, confusion matrix, and classification metrics
5. **Feature Importance** - Top 15 most important features with interactive visualizations
6. **Prediction Panel** - Input customer data to predict churn probability in real-time

All visualizations use Plotly for interactive, production-grade charts.

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the dataset:**
   - Ensure you have the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file
   - Place it in one of these locations:
     - Same directory as `app.py`
     - Or update the path in the `load_and_preprocess_data()` function

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard:**
   - The app will automatically open in your browser
   - Default URL: `http://localhost:8501`

## Features

### Data Processing
- Automatic handling of missing values in TotalCharges
- Label encoding for categorical variables
- SMOTE for handling class imbalance
- Train-test split with 80-20 ratio

### Model
- **Algorithm:** Random Forest Classifier
- **Validation:** 5-fold cross-validation during training
- **Performance Metrics:** Accuracy, Precision, Recall, ROC-AUC

### Interactive Elements
- Real-time filtering by contract type, tenure, and charges
- Dynamic KPI updates based on filters
- Interactive prediction with probability scores
- Responsive layout with wide-screen optimization

## Usage

### Navigation
1. **Sidebar:** Use filters to segment customer data
2. **KPI Section:** View high-level metrics
3. **Insights Section:** Analyze churn patterns
4. **Model Performance:** Evaluate model accuracy
5. **Feature Importance:** Understand key drivers
6. **Prediction Panel:** Test individual customer predictions

### Making Predictions
1. Scroll to the "Customer Churn Prediction" section
2. Enter customer information in the form
3. Click "Predict Churn"
4. View probability and recommendations

## Technical Details

- **Caching:** Uses `@st.cache_data` and `@st.cache_resource` for optimal performance
- **Visualization:** 100% Plotly-based (no matplotlib/seaborn)
- **Layout:** Wide layout with responsive columns
- **Model Training:** Automated on first load, cached for subsequent sessions

## File Structure

```
.
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset (required)
└── README.md                                 # This file
```

## Dependencies

See `requirements.txt` for the complete list. Key packages:
- streamlit: Web application framework
- pandas: Data manipulation
- plotly: Interactive visualizations
- scikit-learn: Machine learning
- imbalanced-learn: SMOTE implementation

## Notes

- The model is trained automatically on app startup
- Training uses SMOTE to handle class imbalance
- All preprocessing matches the original notebook exactly
- No external APIs or services required
- All computations run locally

## Support

For issues or questions:
1. Check that the dataset is in the correct location
2. Verify all dependencies are installed
3. Ensure Python version is 3.8 or higher
