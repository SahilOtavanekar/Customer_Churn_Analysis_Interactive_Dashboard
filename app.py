"""
Customer Churn Analysis Dashboard - Streamlit Application
==========================================================
Production-ready dashboard for analyzing customer churn patterns and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Customer Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 1rem;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING & PREPROCESSING ====================
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the customer churn dataset"""
    try:
        # Try to load from the original path (notebook path)
        df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    except FileNotFoundError:
        try:
            # Try alternative paths
            df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        except FileNotFoundError:
            st.error("Dataset not found. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the correct directory.")
            st.stop()
    
    # Drop customerID column
    df = df.drop(columns=["customerID"])
    
    # Handle TotalCharges - replace empty strings with 0.0
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    
    # Convert Churn to binary
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    
    # Store original data for visualization
    df_original = df.copy()
    
    # Encode categorical variables
    object_columns = df.select_dtypes(include="object").columns
    encoders = {}
    
    for column in object_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        encoders[column] = label_encoder
    
    return df, df_original, encoders

@st.cache_resource
def train_model(df):
    """Train the Random Forest model with SMOTE"""
    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Train Random Forest
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train_smote, y_train_smote)
    
    # Predictions
    y_test_pred = rfc.predict(X_test)
    y_test_proba = rfc.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rfc.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rfc, metrics, feature_importance, X_test, y_test

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df, df_original, encoders = load_and_preprocess_data()
        model, metrics, feature_importance, X_test, y_test = train_model(df)
    
    # ==================== SIDEBAR ====================
    st.sidebar.header("🔍 Filters & Settings")
    
    # Contract Type Filter
    contract_mapping = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
    contract_types = st.sidebar.multiselect(
        "Contract Type",
        options=list(contract_mapping.values()),
        default=list(contract_mapping.values())
    )
    
    # Tenure Range Filter
    tenure_range = st.sidebar.slider(
        "Tenure (months)",
        min_value=int(df_original['tenure'].min()),
        max_value=int(df_original['tenure'].max()),
        value=(int(df_original['tenure'].min()), int(df_original['tenure'].max()))
    )
    
    # Monthly Charges Filter
    monthly_charges_range = st.sidebar.slider(
        "Monthly Charges ($)",
        min_value=float(df_original['MonthlyCharges'].min()),
        max_value=float(df_original['MonthlyCharges'].max()),
        value=(float(df_original['MonthlyCharges'].min()), float(df_original['MonthlyCharges'].max()))
    )
    
    # Apply filters
    reverse_contract_mapping = {v: k for k, v in contract_mapping.items()}
    selected_contract_codes = [reverse_contract_mapping[ct] for ct in contract_types]
    
    filtered_df = df_original[
        (df_original['tenure'] >= tenure_range[0]) &
        (df_original['tenure'] <= tenure_range[1]) &
        (df_original['MonthlyCharges'] >= monthly_charges_range[0]) &
        (df_original['MonthlyCharges'] <= monthly_charges_range[1]) &
        (df_original['Contract'].isin(contract_types))
    ]
    
    # ==================== KPI CARDS ====================
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(filtered_df)
    churn_rate = (filtered_df['Churn'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    retention_rate = 100 - churn_rate
    avg_tenure = filtered_df['tenure'].mean() if len(filtered_df) > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{churn_rate:.2f}%</div>
            <div class="kpi-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{retention_rate:.2f}%</div>
            <div class="kpi-label">Retention Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_tenure:.1f}</div>
            <div class="kpi-label">Avg Tenure (months)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== CHURN INSIGHTS ====================
    st.header("📊 Churn Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn vs Tenure
        tenure_bins = pd.cut(filtered_df['tenure'], bins=[0, 12, 24, 36, 48, 72], labels=['0-12', '13-24', '25-36', '37-48', '49-72'])
        churn_by_tenure = filtered_df.groupby([tenure_bins, 'Churn']).size().unstack(fill_value=0)
        churn_by_tenure = churn_by_tenure.reindex(columns=[0, 1], fill_value=0)
        churn_by_tenure_pct = churn_by_tenure.div(churn_by_tenure.sum(axis=1), axis=0) * 100
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=churn_by_tenure_pct.index.astype(str),
            y=churn_by_tenure_pct[0],
            name='No Churn',
            marker_color='#2ecc71'
        ))
        fig1.add_trace(go.Bar(
            x=churn_by_tenure_pct.index.astype(str),
            y=churn_by_tenure_pct[1],
            name='Churn',
            marker_color='#e74c3c'
        ))
        fig1.update_layout(
            title='Churn Distribution by Tenure',
            xaxis_title='Tenure Range (months)',
            yaxis_title='Percentage (%)',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Churn by Contract Type
        # Create a mapping for churn values
        churn_labels = filtered_df['Churn'].map({0: 'No Churn', 1: 'Churn'})
        churn_by_contract = filtered_df.groupby([filtered_df['Contract'], churn_labels]).size().unstack(fill_value=0)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=churn_by_contract.index,
            y=churn_by_contract.get('No Churn', [0]*len(churn_by_contract)),
            name='No Churn',
            marker_color='#2ecc71'
        ))
        fig2.add_trace(go.Bar(
            x=churn_by_contract.index,
            y=churn_by_contract.get('Churn', [0]*len(churn_by_contract)),
            name='Churn',
            marker_color='#e74c3c'
        ))
        fig2.update_layout(
            title='Churn Distribution by Contract Type',
            xaxis_title='Contract Type',
            yaxis_title='Count',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly Charges vs Churn
    fig3 = px.box(
        filtered_df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        labels={'Churn': 'Churn Status', 'MonthlyCharges': 'Monthly Charges ($)'},
        title='Monthly Charges Distribution by Churn Status',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
    )
    fig3.update_xaxes(tickvals=[0, 1], ticktext=['No Churn', 'Churn'])
    fig3.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    # ==================== MODEL PERFORMANCE ====================
    st.header("🎯 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True
        ))
        fig_cm.update_layout(
            title='Confusion Matrix',
            height=400,
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Classification Report
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_df = report_df.iloc[:-3, :-1]  # Remove accuracy, macro avg, weighted avg rows and support column
        
        fig_report = go.Figure()
        fig_report.add_trace(go.Bar(
            x=report_df.index,
            y=report_df['precision'],
            name='Precision',
            marker_color='#3498db'
        ))
        fig_report.add_trace(go.Bar(
            x=report_df.index,
            y=report_df['recall'],
            name='Recall',
            marker_color='#e74c3c'
        ))
        fig_report.add_trace(go.Bar(
            x=report_df.index,
            y=report_df['f1-score'],
            name='F1-Score',
            marker_color='#2ecc71'
        ))
        fig_report.update_layout(
            title='Classification Metrics by Class',
            xaxis_title='Class',
            yaxis_title='Score',
            barmode='group',
            height=400,
            xaxis=dict(tickvals=[0, 1], ticktext=['No Churn', 'Churn'])
        )
        st.plotly_chart(fig_report, use_container_width=True)
    
    # ==================== FEATURE IMPORTANCE ====================
    st.header("🔍 Feature Importance & Explainability")
    
    # Top 15 features
    top_features = feature_importance.head(15)
    
    fig_importance = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 15 Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Feature Importance Table
    with st.expander("📋 View Complete Feature Importance Table"):
        st.dataframe(
            feature_importance.style.format({'importance': '{:.4f}'}),
            use_container_width=True,
            height=400
        )
    
    # ==================== PREDICTION PANEL ====================
    st.header("🔮 Customer Churn Prediction")
    
    st.markdown("### Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                      ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=100.0, step=0.01)
    
    if st.button("🔮 Predict Churn", type="primary"):
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[column])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **High Risk of Churn**")
                st.markdown(f"**Churn Probability:** {pred_proba[1]:.2%}")
            else:
                st.success("✅ **Low Risk of Churn**")
                st.markdown(f"**Retention Probability:** {pred_proba[0]:.2%}")
        
        with col2:
            # Probability gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_proba[1] * 100,
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if prediction == 1 else "#2ecc71"},
                    'steps': [
                        {'range': [0, 33], 'color': "#d5f4e6"},
                        {'range': [33, 66], 'color': "#fff9db"},
                        {'range': [66, 100], 'color': "#fadbd8"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if prediction == 1:
            st.markdown("""
            - **Immediate Action Required:** Contact customer to understand concerns
            - **Offer Incentives:** Consider promotional offers or discounts
            - **Improve Service:** Review service quality and address pain points
            - **Contract Upgrade:** Encourage longer-term contract commitments
            """)
        else:
            st.markdown("""
            - **Maintain Engagement:** Continue providing excellent service
            - **Upsell Opportunities:** Consider offering additional services
            - **Loyalty Rewards:** Implement rewards program for long-term customers
            """)

if __name__ == "__main__":
    main()
