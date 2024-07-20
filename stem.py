import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# Load the CSV file into a DataFrame
df = pd.read_csv('drug200.csv')  # Replace 'drug200.csv' with your actual CSV file path

# Split into X (features) and y (target variable)
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']

st.title("Drug predictor")

# Preprocess the data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Sex', 'BP', 'Cholesterol'])], remainder='passthrough')
X = ct.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

# Train the model
regressor = RandomForestClassifier(n_estimators=10, random_state=0)
regressor.fit(X, y)

def get_user_input_and_predict():
    age = st.number_input("Enter age: ", min_value=0, max_value=120)
    sex = st.selectbox("Enter sex (F/M):", ["F", "M"])
    bp = st.selectbox("Enter BP (LOW/NORMAL/HIGH):", ["LOW", "NORMAL", "HIGH"])
    cholesterol = st.selectbox("Enter cholesterol (NORMAL/HIGH):", ["NORMAL", "HIGH"])
    na_to_k = st.number_input("Enter Na_to_K ratio: ")

    # Create input array matching the transformed format
    input_data = pd.DataFrame([[age, sex, bp, cholesterol, na_to_k]], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
    input_transformed = ct.transform(input_data)

    # Predict using the trained regressor
    predicted_value = regressor.predict(input_transformed).astype(int)

    # Inverse transform to decode back to original labels
    predicted_label = le.inverse_transform(predicted_value)

    st.write("Predicted drug:", predicted_label[0])

# Call the function to get user input and predict
get_user_input_and_predict()
