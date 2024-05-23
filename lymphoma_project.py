#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ucimlrepo import fetch_ucirepo 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch dataset
lymphography = fetch_ucirepo(id=63)

# Data (as pandas dataframes)
X = lymphography.data.features
y = lymphography.data.targets['class'] - 1  # Adjust class labels to start from 0

# Ensure column names are unique
column_names = ["lymphatics", "block of affere", "bl. of lymph. c", "bl. of lymph. s", "by pass", "extravasates", 
                "regeneration of", "early uptake in", "lym.nodes dimin", "lym.nodes enlar", "changes in lym", 
                "defect in node", "changes in node 1", "changes in node 2", "changes in stru", "special forms", 
                "dislocation of", "exclusion of no", "no. of nodes in"]
X.columns = column_names

# Metadata
metadata = lymphography.metadata
variables = lymphography.variables

# Page Configuration
st.set_page_config(page_title="Lymphoma Cancer Outcome Prediction Dashboard", layout="wide")

# Sidebar for user input
st.sidebar.header("User Input Features")
st.sidebar.write("Please enter values for the following features:")

def user_input_features():
    user_data = {}
    for column in column_names:
        user_data[column] = st.sidebar.number_input(column, value=0)
    return pd.DataFrame(user_data, index=[0])

input_df = user_input_features()

# Main Page
st.title("Lymphoma Cancer Outcome Prediction Dashboard")

# Data Exploration Section
st.header("1. Data Exploration")
st.write("### Dataset Metadata")
st.write(metadata)
st.write("### Variable Information")
st.write(variables)
st.write("### Data Sample")
st.dataframe(X.head())

# Feature Engineering Section
st.header("2. Feature Engineering")
st.write("### Scaled Features")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
st.dataframe(X_scaled.head())

# Model Development Section
st.header("3. Model Development and Evaluation")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# ROC Curve
st.write("### ROC Curve")
y_test_bin = pd.get_dummies(y_test)
y_pred_prob = model.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}

fig, ax = plt.subplots()
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin.iloc[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    ax.plot(fpr[i], tpr[i], label=f'ROC curve class {i} (area = {roc_auc[i]:.2f})')

ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
st.pyplot(fig)

# Real-time Prediction Section
st.header("4. Real-time Prediction")
st.write("### Enter values for prediction")
st.dataframe(input_df)

if st.button("Predict"):
    new_data_scaled = scaler.transform(input_df)
    prediction = model.predict(new_data_scaled)
    st.write(f"### Prediction: {prediction[0]}")
    prediction_proba = model.predict_proba(new_data_scaled)
    st.write(f"### Prediction Probabilities: {prediction_proba}")

# Footer
st.write("Developed by Sean Farquharson")




