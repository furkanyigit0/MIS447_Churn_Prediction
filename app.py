import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Customer Churn Early Warning System", layout="wide")

# 1. LOAD THE MODEL AND FEATURES
# Load the saved Random Forest model
try:
    model = joblib.load('random_forest_churn_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'random_forest_churn_model.pkl' not found. Ensure it was saved in Block 5.")
    st.stop()

# Load feature names (These were saved implicitly from X_train columns)
# Note: In a real-world scenario, you would explicitly save these features.
# For simplicity, we define the required input columns based on the original dataset structure.

# List of all features needed for the model (derived from the original dataset and encoding)
# This MUST match the features the model was trained on!
# We manually define the categorical options used for one-hot encoding (drop_first=True)
features_list = [
    # Numerical
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    # Binary Features (1 or 0)
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    # Multi-class Features (One-Hot Encoded)
    'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
    'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]


# Function to collect user input
def get_user_input():
    st.sidebar.header('Customer Profile Input')

    # Numerical Inputs
    tenure = st.sidebar.slider('Tenure (Months)', 1, 72, 24)
    MonthlyCharges = st.sidebar.slider('Monthly Charges ($)', 18.25, 118.75, 50.0)
    TotalCharges = st.sidebar.slider('Total Charges ($)', 18.8, 8684.8, 1000.0)

    # Binary Inputs
    gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', (0, 1))
    Partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    PhoneService = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))

    # Multi-class Inputs
    InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    Contract = st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    PaymentMethod = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

    # Additional Services (simplified as Yes/No/No internet service)
    def select_service(label):
        return st.sidebar.selectbox(label, ('Yes', 'No', 'No internet service'))

    OnlineSecurity = select_service('Online Security')
    OnlineBackup = select_service('Online Backup')
    DeviceProtection = select_service('Device Protection')
    TechSupport = select_service('Tech Support')
    StreamingTV = select_service('Streaming TV')
    StreamingMovies = select_service('Streaming Movies')

    # Create a dictionary for user data based on Streamlit inputs
    user_data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'SeniorCitizen': SeniorCitizen,
        # Categorical features are handled later via one-hot encoding logic
        'gender': gender,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'PaperlessBilling': PaperlessBilling,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
    }

    return pd.DataFrame(user_data, index=[0])


# 2. RUN THE PREDICTION
if 'model' in locals():
    # Get user data
    input_df = get_user_input()

    st.subheader('1. User Profile:')
    st.write(input_df)

    # --- ENCODING INPUT DATA ---
    # Convert user input to the exact format expected by the trained model (One-Hot Encoded)

    # Create a DataFrame with all features used during training, initially set to 0
    final_features_df = pd.DataFrame(0, index=[0], columns=features_list)

    # Map raw input to the encoded columns (matching the model training logic)

    # Simple Binary Features
    if input_df['gender'].iloc[0] == 'Male':
        final_features_df['gender_Male'] = 1

    for feature in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if input_df[feature].iloc[0] == 'Yes':
            final_features_df[f'{feature}_Yes'] = 1

    # Numeric Features
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
    # DSL is the base case (encoded as 0 for both 'Fiber optic' and 'No')

    # Services (OnlineSecurity, OnlineBackup, etc.)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for svc in service_cols:
        if input_df[svc].iloc[0] == 'Yes':
            final_features_df[f'{svc}_Yes'] = 1
        elif input_df[svc].iloc[0] == 'No internet service':
             # If "No internet service", the two encoded columns ('svc_Yes' and 'svc_No internet service') are both 0 in the training data,
             # so no change is needed from the initial 0 state.
             pass

    # Contract
    if input_df['Contract'].iloc[0] == 'One year':
        final_features_df['Contract_One year'] = 1
    elif input_df['Contract'].iloc[0] == 'Two year':
        final_features_df['Contract_Two year'] = 1
    # 'Month-to-month' is the base case (encoded as 0 for both)

    # Payment Method
    if input_df['PaymentMethod'].iloc[0] == 'Credit card (automatic)':
        final_features_df['PaymentMethod_Credit card (automatic)'] = 1
    elif input_df['PaymentMethod'].iloc[0] == 'Electronic check':
        final_features_df['PaymentMethod_Electronic check'] = 1
    elif input_df['PaymentMethod'].iloc[0] == 'Mailed check':
        final_features_df['PaymentMethod_Mailed check'] = 1
    # 'Bank transfer (automatic)' is the base case (encoded as 0 for all three)


    st.subheader('2. Encoded Input for Model:')
    st.write(final_features_df)
    final_features_df = final_features_df[features_list]
    # 3. PREDICTION
    prediction = model.predict(final_features_df)
    prediction_proba = model.predict_proba(final_features_df)

    st.subheader('3. Prediction:')

    churn_status = "Customer is LIKELY to Churn (High Risk)" if prediction[0] == 1 else "Customer is NOT Likely to Churn (Low Risk)"

    if prediction[0] == 1:
        st.error(churn_status)
    else:
        st.success(churn_status)

    st.write(f"Churn Probability (Risk Score): **{prediction_proba[0][1] * 100:.2f}%**")
    st.write(f"No Churn Probability: **{prediction_proba[0][0] * 100:.2f}%**")
