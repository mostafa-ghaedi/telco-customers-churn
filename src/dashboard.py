import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Streamlit app configuration
st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide")

# Streamlit app
st.title("Telco Customer Churn Dashboard")

# Load dataset
try:
    df_telco = pd.read_csv("telco_churn.csv")
except FileNotFoundError:
    st.error("Please ensure 'telco_churn.csv' is in the same directory as this script.")
    st.stop()

# Preprocess data for visualization (keep original for model)
df_viz = df_telco.copy()
df_viz['TotalCharges'] = pd.to_numeric(df_viz['TotalCharges'], errors='coerce')
df_viz.dropna(inplace=True)
df_viz.drop(columns='customerID', inplace=True)
df_viz["PaymentMethod"] = df_viz["PaymentMethod"].str.replace(" (automatic)", '', regex=False)

# Sidebar for filters
st.sidebar.header("Filters")
selected_gender = st.sidebar.selectbox("Select Gender", ['All'] + list(df_viz['gender'].unique()))
selected_contract = st.sidebar.selectbox("Select Contract", ['All'] + list(df_viz['Contract'].unique()))
tenure_range = st.sidebar.slider("Select Tenure Range",
                                 min_value=int(df_viz['tenure'].min()),
                                 max_value=int(df_viz['tenure'].max()),
                                 value=(int(df_viz['tenure'].min()), int(df_viz['tenure'].max())))
monthly_charges_range = st.sidebar.slider("Select Monthly Charges Range",
                                         min_value=int(df_viz['MonthlyCharges'].min()),
                                         max_value=int(df_viz['MonthlyCharges'].max()),
                                         value=(int(df_viz['MonthlyCharges'].min()), int(df_viz['MonthlyCharges'].max())))

# Apply filters for visualization
filtered_df = df_viz.copy()
if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
if selected_contract != 'All':
    filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
filtered_df = filtered_df[(filtered_df['tenure'] >= tenure_range[0]) & (filtered_df['tenure'] <= tenure_range[1])]
filtered_df = filtered_df[(filtered_df['MonthlyCharges'] >= monthly_charges_range[0]) & 
                         (filtered_df['MonthlyCharges'] <= monthly_charges_range[1])]

# Dataset Overview
st.header("Dataset Overview")
st.write(f"Number of rows after filtering: {filtered_df.shape[0]}")
st.write(f"Number of columns: {filtered_df.shape[1]}")
st.dataframe(filtered_df.head())

# Churn Distribution
st.header("Churn Distribution")
prop_response = filtered_df["Churn"].value_counts(normalize=True).reset_index()
prop_response.columns = ['Churn', 'Proportion']
st.bar_chart(prop_response.set_index('Churn'))

# Demographic Information
st.header("Demographic Information")
demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
selected_demo = st.selectbox("Select Demographic Feature", demographic_columns)
prop_demo = pd.crosstab(filtered_df[selected_demo], filtered_df['Churn']).apply(
    lambda x: x / x.sum() * 100, axis=1
).reset_index()
prop_demo.columns = [selected_demo, 'No', 'Yes']
st.bar_chart(prop_demo.set_index(selected_demo))

# Customer Account Information
st.header("Customer Account Information")
account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']
selected_account = st.selectbox("Select Account Feature", account_columns)
prop_account = pd.crosstab(filtered_df[selected_account], filtered_df['Churn']).apply(
    lambda x: x / x.sum() * 100, axis=1
).reset_index()
prop_account.columns = [selected_account, 'No', 'Yes']
st.bar_chart(prop_account.set_index(selected_account))

# Numeric Customer Account Information
st.header("Numeric Customer Account Information")
account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
selected_numeric = st.selectbox("Select Numeric Feature", account_columns_numeric)
bins = 10
hist_data = filtered_df[[selected_numeric, 'Churn']].copy()
hist_data['Bin'] = pd.cut(hist_data[selected_numeric], bins=bins)
hist_data['Bin_Midpoint'] = hist_data['Bin'].apply(lambda x: x.mid).astype(float)
hist_pivot = hist_data.pivot_table(index='Bin_Midpoint', columns='Churn', aggfunc='size', fill_value=0)
hist_pivot = hist_pivot.div(hist_pivot.sum(axis=1), axis=0) * 100  # Normalize to percentage
hist_pivot.reset_index(inplace=True)
st.bar_chart(hist_pivot.set_index('Bin_Midpoint'))

# Services Information
st.header("Services Information")
services_columns = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
selected_service = st.selectbox("Select Service Feature", services_columns)
prop_service = pd.crosstab(filtered_df[selected_service], filtered_df['Churn']).apply(
    lambda x: x / x.sum() * 100, axis=1
).reset_index()
prop_service.columns = [selected_service, 'No', 'Yes']
st.bar_chart(prop_service.set_index(selected_service))

# Model Training and Evaluation (using original unfiltered data)
st.header("Model Performance")
df_telco_transformed = df_telco.copy()

# Preprocess data for model (same as original code)
df_telco_transformed['TotalCharges'] = pd.to_numeric(df_telco_transformed['TotalCharges'], errors='coerce')
df_telco_transformed.dropna(inplace=True)
df_telco_transformed.drop(columns='customerID', inplace=True)
df_telco_transformed["PaymentMethod"] = df_telco_transformed["PaymentMethod"].str.replace(" (automatic)", '', regex=False)

# Label encoding
label_encoding_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']
for column in label_encoding_columns:
    if column == 'gender':
        df_telco_transformed[column] = df_telco_transformed[column].map({'Female': 1, 'Male': 0})
    else:
        df_telco_transformed[column] = df_telco_transformed[column].map({'Yes': 1, 'No': 0})

# One-hot encoding
one_hot_encoding_columns = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]
df_telco_transformed = pd.get_dummies(df_telco_transformed, columns=one_hot_encoding_columns)

# Min-max normalization
min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
for column in min_max_columns:
    min_column = df_telco_transformed[column].min()
    max_column = df_telco_transformed[column].max()
    df_telco_transformed[column] = (df_telco_transformed[column] - min_column) / (max_column - min_column)

# Split data
X = df_telco_transformed.drop(columns="Churn")
y = df_telco_transformed["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40, shuffle=True)

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=2)
gb_model.fit(X_train, y_train)
predictions = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Display model accuracy
st.write(f"Gradient Boosting Classifier Accuracy: {accuracy:.4f}")

# Feature Importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)  # Top 10 features
st.bar_chart(feature_importance.set_index('Feature'))