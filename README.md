
# Lymphoma Cancer Outcome Prediction Dashboard

This project aims to create an interactive dashboard for predicting lymphoma cancer outcomes using machine learning. The dashboard is built with Python and Streamlit, utilizing the Lymphography dataset from the UCI Machine Learning Repository.

## Features

- **Data Exploration**: Display metadata, variable information, and sample data.
- **Feature Engineering**: Standardize features and show scaled data.
- **Model Development**: Train an XGBoost classifier to predict lymphoma outcomes and evaluate model performance with accuracy, confusion matrix, and ROC curve.
- **Real-time Prediction**: Allow users to input feature values and get real-time predictions.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- ucimlrepo

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/lymphoma-dashboard.git
cd lymphoma-dashboard
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Dashboard

1. Run the Streamlit app:

```bash
streamlit run lymphoma_dashboard.py
```

2. Open your web browser and go to `http://localhost:8501` to view the dashboard.

## Dashboard Sections

1. **Data Exploration**
   - Displays dataset metadata and variable information.
   - Shows a sample of the dataset.

2. **Feature Engineering**
   - Standardizes the features.
   - Displays the scaled features.

3. **Model Development and Evaluation**
   - Trains an XGBoost model on the dataset.
   - Displays model accuracy, confusion matrix, and ROC curve.

4. **Real-time Prediction**
   - Provides user input fields for feature values.
   - Displays prediction and prediction probabilities based on user input.

## Dataset

The Lymphography dataset is sourced from the UCI Machine Learning Repository. It includes various features related to lymphatic diagnosis and is used to predict the type of lymphoma.

## Authors

- [Your Name](https://github.com/yourusername)

## License

This project is licensed under the MIT License.
