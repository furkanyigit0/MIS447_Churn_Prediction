import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- INITIAL SETUP ---
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.title("📞 Customer Churn Early Warning System")
st.markdown("---")


# 1. LOAD THE MODEL AND FEATURES
# Load the saved Random Forest model and feature list (from model_features.json)
try:
    # Load model
    model = joblib.load('random_forest_churn_model.pkl')
    # Load expected feature list and order
    with open('model_features.json', 'r') as f:
        features_list = json.load(f)
    
    # Load feature importances for visualization
    rf_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features_list, 'importance': rf_importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).set_index('feature')

except FileNotFoundError:
    st.error("Error: Required model files ('random_forest_churn_model.pkl' or 'model_features.json') not found. Please check your GitHub repository.")
    st.stop()


# 2. INPUT COLLECTION (New Layout for Differentiation)
st.header('1. Define Customer Profile')

# Use st.columns to create a new layout (Different from the single sidebar layout)
col1, col2, col3 = st.columns(3)

# Function to collect user input (moved to main body)
def get_user_input(col1, col2, col3):
    
    # --- COLUMN 1: Tenure & Demographics ---
    with col1:
        st.subheader("Demographics & Tenure")
        tenure = st.slider('Tenure (Months)', 1, 72, 12, help="How long the customer has been with the company.")
        SeniorCitizen = st.selectbox('Senior Citizen', (0, 1), help="1 if customer is 65 or older, 0 otherwise.")
        gender = st.selectbox('Gender', ('Female', 'Male'))
        Partner = st.selectbox('Partner', ('Yes', 'No'))
        Dependents = st.selectbox('Dependents', ('Yes', 'No'))
        
    # --- COLUMN 2: Charges & Contract ---
    with col2:
        st.subheader("Billing & Contract")
        MonthlyCharges = st.slider('Monthly Charges ($)', 18.25, 118.75, 50.0)
        TotalCharges = st.slider('Total Charges ($)', 18.8, 8684.8, 500.0)
        Contract = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
        PaperlessBilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        
    # --- COLUMN 3: Services ---
    with col3:
        st.subheader("Services Subscribed")
        PhoneService = st.selectbox('Phone Service', ('Yes', 'No'))
        MultipleLines = st.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
        InternetService = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
        
        # Additional Services (simplified)
        def select_service(label):
            return st.selectbox(label, ('Yes', 'No', 'No internet service'))

        OnlineSecurity = select_service('Online Security')
        OnlineBackup = select_service('Online Backup')
        TechSupport = select_service('Tech Support')
        DeviceProtection = select_service('Device Protection')
        StreamingTV = select_service('Streaming TV')
        StreamingMovies = select_service('Streaming Movies')

    # Create a dictionary for user data
    user_data = {
        'tenure': tenure, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges, 
        'SeniorCitizen': SeniorCitizen, 'gender': gender, 'Partner': Partner, 'Dependents': Dependents,
        'PhoneService': PhoneService, 'PaperlessBilling': PaperlessBilling, 'MultipleLines': MultipleLines,
        'InternetService': InternetService, 'Contract': Contract, 'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup, 'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
        'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
    }
    
    # Add Payment Method outside columns for simpler layout
    PaymentMethod = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    user_data['PaymentMethod'] = PaymentMethod

    return pd.DataFrame(user_data, index=[0])


# Collect input and show the input data
input_df = get_user_input(col1, col2, col3)
st.subheader('2. Review Encoded Input:')


# --- 3. ENCODING INPUT DATA ---
# Convert user input to the exact format expected by the trained model (One-Hot Encoded)

# Create a DataFrame with all features used during training, initially set to 0
final_features_df = pd.DataFrame(0, index=[0], columns=features_list)

# Map raw input to the encoded columns (matching the model training logic)

# Simple Binary Features and Numerics
if input_df['gender'].iloc[0] == 'Male':
    final_features_df['gender_Male'] = 1

for feature in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    if input_df[feature].iloc[0] == 'Yes':
        final_features_df[f'{feature}_Yes'] = 1

# Numeric Features (already handled as numerics, just assign)
final_features_df['SeniorCitizen'] = input_df['SeniorCitizen'].iloc[0]
final_features_df['tenure'] = input_df['tenure'].iloc[0]
final_features_df['MonthlyCharges'] = input_df['MonthlyCharges'].iloc[0]
final_features_df['TotalCharges'] = input_df['TotalCharges'].iloc[0]

# Multi-class and complex features (mapping the user selection to the encoded columns)

# MultipleLines
if input_df['MultipleLines'].iloc[0] == 'Yes':
    final_features_df['MultipleLines_Yes'] = 1

# Internet Service
if input_df['InternetService'].iloc[0] == 'Fiber optic':
    final_features_df['InternetService_Fiber optic'] = 1
elif input_df['InternetService'].iloc[0] == 'No':
    final_features_df['InternetService_No'] = 1
# DSL is the base case

# Services (OnlineSecurity, OnlineBackup, etc.)
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for svc in service_cols:
    if input_df[svc].iloc[0] == 'Yes':
        final_features_df[f'{svc}_Yes'] = 1
    # 'No internet service' is implicitly handled by setting both _Yes and _NointernetService to 0

# Contract
if input_df['Contract'].iloc[0] == 'One year':
    final_features_df['Contract_One year'] = 1
elif input_df['Contract'].iloc[0] == 'Two year':
    final_features_df['Contract_Two year'] = 1
# 'Month-to-month' is the base case

# Payment Method
if input_df['PaymentMethod'].iloc[0] == 'Credit card (automatic)':
    final_features_df['PaymentMethod_Credit card (automatic)'] = 1
elif input_df['PaymentMethod'].iloc[0] == 'Electronic check':
    final_features_df['PaymentMethod_Electronic check'] = 1
elif input_df['PaymentMethod'].iloc[0] == 'Mailed check':
    final_features_df['PaymentMethod_Mailed check'] = 1
# 'Bank transfer (automatic)' is the base case


# --- CRITICAL FIX: Ensure the final dataframe has the exact columns and order the model expects ---
final_features_df = final_features_df[features_list]

# Display the final encoded input
st.dataframe(final_features_df)


# --- 4. PREDICTION ---
st.header('3. Prediction Result')

if st.button('Predict Churn Risk'):
    prediction = model.predict(final_features_df)
    prediction_proba = model.predict_proba(final_features_df)
    
    risk_score = prediction_proba[0][1] * 100
    
    st.subheader(f"Risk Score: {risk_score:.2f}%")

    if prediction[0] == 1:
        st.error(f"Customer is **LIKELY to Churn** (High Risk: {risk_score:.2f}%)")
        st.markdown("---")
    else:
        st.success(f"Customer is **NOT Likely to Churn** (Low Risk: {100 - risk_score:.2f}%)")
        st.balloons()
        st.markdown("---")

    # --- DIFFERENCE VISUALIZATION (Extra Feature) ---
    st.subheader("Feature Importance Overview")
    st.markdown("Factors are ranked by how much they generally influence the model's decision.")

    # Show Feature Importance Plot (New Element for Differentiation)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=feature_importance_df['importance'][:10], y=feature_importance_df.index[:10], ax=ax, palette='viridis')
    ax.set_title('Top 10 Most Important Churn Drivers')
    ax.set_xlabel('Relative Importance Score')
    ax.set_ylabel('Feature')
    st.pyplot(fig)
