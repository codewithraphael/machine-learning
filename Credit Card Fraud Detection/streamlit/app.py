import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💳 Credit Card Fraud Detection System")

# Sidebar for model selection
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
    help="Choose the ML model for prediction"
)

# Load the selected model
models_path = {
    "Logistic Regression": BASE_DIR.parent / "models"/"logistic_regression_model.joblib",
    "Decision Tree": BASE_DIR.parent / "models"/"decision_tree_classifier_model.joblib",
    "Random Forest": BASE_DIR.parent / "models"/"randomforest_classifier_model.joblib",
    "XGBoost": BASE_DIR.parent / "models"/"xgboost_classifier_model.joblib"
}

try:
    model = joblib.load(models_path[model_choice])
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Main interface
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Feature Guide"])

with tab1:
    st.header("Transaction Analysis")
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Transaction Details")
        amount = st.number_input(
            "💰 Transaction Amount ($)",
            min_value=0.0,
            max_value=100000.0,
            value=500.0,
            step=10.0
        )
        
        # Use the same merchant category labels as in the training dataset
        merchant_options = [
            "Fuel",
            "Grocery",
            "Online_Shopping",
            "Restaurant",
            "Entertainment",
            "Healthcare",
            "Travel"
        ]
        merchant_category = st.selectbox(
            "🏪 Merchant Category",
            merchant_options
        )
        
        transaction_hour = st.slider(
            "🕐 Transaction Hour",
            min_value=0,
            max_value=23,
            value=12,
            help="Hour of day (0-23)"
        )
    
    with col2:
        st.subheader("Customer Information")
        customer_age = st.slider(
            "👤 Customer Age",
            min_value=18,
            max_value=80,
            value=40,
            help="Customer's age in years"
        )
        
        account_balance = st.number_input(
            "💵 Account Balance ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=5000.0,
            step=100.0
        )
    
    with col3:
        st.subheader("Transaction Context")
        is_foreign = st.checkbox(
            "🌍 International Transaction",
            value=False
        )
        
        is_weekend = st.checkbox(
            "📅 Weekend Transaction",
            value=False
        )
        
        num_transactions = st.number_input(
            "📈 Transactions Today",
            min_value=0,
            max_value=50,
            value=1,
            help="Number of transactions made today"
        )
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
        # Prepare features for model as a DataFrame with the same column names
        # used during training. This ensures the ColumnTransformer receives
        # the expected columns and order.
        features_df = pd.DataFrame(
            [{
                "amount": amount,
                "merchant_category": merchant_category,
                "transaction_hour": transaction_hour,
                "customer_age": customer_age,
                "account_balance": account_balance,
                "is_foreign_transaction": int(is_foreign),
                "is_weekend": int(is_weekend),
                "num_transactions_today": int(num_transactions)
            }]
        )
        
        # Make prediction
        try:
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0]
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 0:
                    st.success("✅ **LEGITIMATE TRANSACTION**", icon="✅")
                    st.markdown("""
                    This transaction appears to be genuine based on the model's analysis.
                    """)
                else:
                    st.error("⚠️ **FRAUDULENT TRANSACTION DETECTED**", icon="⚠️")
                    st.markdown("""
                    This transaction shows signs of fraud. Consider:
                    - Contacting the customer
                    - Temporary account freeze
                    - Additional verification
                    """)
            
            with col2:
                st.metric(
                    label="Fraud Probability",
                    value=f"{probability[1]*100:.2f}%",
                    delta=f"Legitimate: {probability[0]*100:.2f}%"
                )
            
            # Detailed breakdown
            st.markdown("---")
            st.subheader("📋 Detailed Analysis")
            
            analysis_data = {
                "Feature": [
                    "Amount",
                    "Merchant Category",
                    "Hour",
                    "Customer Age",
                    "Account Balance",
                    "Foreign Transaction",
                    "Weekend Transaction",
                    "Daily Transactions"
                ],
                "Value": [
                    f"${amount:.2f}",
                    f"{merchant_category}",
                    f"{transaction_hour}:00",
                    f"{customer_age} years",
                    f"${account_balance:.2f}",
                    "Yes" if is_foreign else "No",
                    "Yes" if is_weekend else "No",
                    f"{num_transactions}"
                ]
            }
            
            st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
            
            # Risk indicators
            st.markdown("---")
            st.subheader("⚠️ Risk Factors")
            
            risk_factors = []
            if amount > 5000:
                risk_factors.append("🔴 High transaction amount")
            if is_foreign:
                risk_factors.append("🟡 International transaction")
            if transaction_hour < 6 or transaction_hour > 22:
                risk_factors.append("🟡 Unusual hour (late night/early morning)")
            if num_transactions > 5:
                risk_factors.append("🟡 Multiple transactions today")
            if account_balance < amount:
                risk_factors.append("🔴 Transaction exceeds account balance")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.write("✅ No major risk factors detected")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

with tab2:
    st.header("📊 Feature Guide")
    
    features_info = {
        "💰 Amount": "The transaction amount in dollars. Higher amounts may be flagged as suspicious.",
        "🕐 Transaction Hour": "Hour of day when transaction occurs (0-23). Late-night transactions may be flagged.",
        "👤 Customer Age": "Age of the cardholder in years. Used to identify unusual usage patterns.",
        "💵 Account Balance": "Current balance in the customer's account. Helps detect insufficient funds.",
        "🌍 International": "Whether transaction is from outside the country. May indicate fraud.",
        "📅 Weekend": "Whether transaction occurs on weekend. Some fraud patterns differ by day.",
        "📈 Daily Transactions": "Count of transactions made by customer today. Helps detect velocity fraud."
    }
    
    for feature, description in features_info.items():
        st.write(f"**{feature}**: {description}")
    
    st.markdown("---")
    st.subheader("Model Information")
    st.write(f"**Currently Selected Model**: {model_choice}")
    st.write("""
    The model is trained on historical transaction data to identify patterns associated with fraud.
    It considers multiple factors to provide a fraud probability score.
    """)
