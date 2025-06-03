# Telco Customer Churn Prediction

## Project Overview
This project predicts customer churn for a telecommunications company using the Telco Customer Churn dataset. The goal is to identify customers likely to leave, enabling targeted retention strategies. The project includes data preprocessing, visualization, and training a Gradient Boosting Classifier. An interactive Streamlit dashboard, built entirely with Streamlit's native charting tools, visualizes data distributions, model performance, and feature importance with user-defined filters. The model achieves an accuracy of approximately 0.79, consistent with the original implementation.

## Dataset
The dataset used is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle, containing 7043 rows and 21 columns, including customer demographics, account information, services, and the target variable `Churn`.

## Features
- **Data Preprocessing**: Cleaning, label and one-hot encoding for categorical variables, and min-max normalization for numerical features.
- **Data Visualization**: Interactive bar charts for categorical features and numerical feature distributions using bin midpoints, all built with Streamlit's `st.bar_chart`.
- **Interactive Filters**: Filters for gender, contract type, tenure range, and monthly charges range to dynamically explore data subsets for visualizations.
- **Model Training**: A Gradient Boosting Classifier trained on the full dataset to predict churn, achieving ~0.79 accuracy, with feature importance displayed.
- **Streamlit Dashboard**: An interactive dashboard with bar charts for all visualizations, implemented without Matplotlib.

## Requirements
To run this project, install the following Python packages:
- pandas
- scikit-learn
- streamlit

Install the dependencies using:
```bash
pip install pandas scikit-learn streamlit
```

## Installation
1. Clone the repository or download the project files.
2. Download the `telco_churn.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project directory.
3. Install the required packages (see above).

## Usage
1. Run the preprocessing and modeling script (`preprocess_and_model.py`) to process the data and train the model:
   ```bash
   python preprocess_and_model.py
   ```
2. Launch the Streamlit dashboard to explore the data and results:
   ```bash
   streamlit run streamlit_dashboard.py
   ```
3. Open the provided URL in your browser to interact with the dashboard.

## Dashboard Features
- **Filters**: Select gender, contract type, tenure range, and monthly charges range to filter data for visualizations.
- **Churn Distribution**: Bar chart showing the proportion of churned vs. non-churned customers.
- **Demographic, Account, and Service Visualizations**: Interactive bar charts for categorical features (e.g., gender, contract).
- **Numeric Visualizations**: Bar charts for numerical feature distributions (e.g., tenure, monthly charges) using bin midpoints.
- **Model Performance**: Displays the Gradient Boosting Classifier's accuracy (~0.79) and a bar chart of the top 10 feature importances.

## Files
- `telco_churn.csv`: The dataset file (download from Kaggle).
- `preprocess_and_model.py`: Script for data preprocessing and model training.
- `streamlit_dashboard.py`: Script for the interactive Streamlit dashboard.
- `README.md`: This documentation file.

## Contact
For questions or feedback, feel free to reach out via [mustafa.ghaedi@gmail.com].