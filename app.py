import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan preprocessor
model = joblib.load('XGBoost.joblib')
preprocessor = joblib.load('preprocessor.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# UI Judul
st.title("Student Dropout Prediction")

# Input Form
st.header("Masukkan Data Mahasiswa")

age = st.number_input("Age", min_value=15, max_value=80, value=20)
marital_status = st.selectbox("Marital Status", ['single', 'married', 'divorced', 'widower', 'facto union', 'legally separated'])
course = st.selectbox("Course", [
    'Biofuel Production Technologies', 'Animation and Multimedia Design', 'Social Service (evening attendance)', 
    'Journalism and Communication', 'Advertising and Marketing Management', 'Tourism', 'Basic Education',
    'Management (evening attendance)', 'Communication Design', 'Social Service', 'Management', 'Law',
    'Marketing', 'Informatics Engineering', 'Journalism', 'Equinculture', 'Nursing'
])
nacionality = st.selectbox("Nacionality", [
    'Portuguese', 'German', 'Spanish', 'Italian', 'Romanian', 'Brazilian', 'Ukrainian', 'Russian',
    'English', 'Lithuanian', 'Cuban', 'Colombian'
])

displaced = st.selectbox("Displaced", ['Yes', 'No'])
educational_special_needs = st.selectbox("Special Needs", ['Yes', 'No'])
debtor = st.selectbox("Debtor", ['Yes', 'No'])
tuition_up_to_date = st.selectbox("Tuition up to date", ['Yes', 'No'])
gender = st.selectbox("Gender", ['Male', 'Female'])
scholarship_holder = st.selectbox("Scholarship Holder", ['Yes', 'No'])
international = st.selectbox("International", ['Yes', 'No'])

# Mapping input binary ke 0/1
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

# Buat dataframe dari input
input_data = pd.DataFrame([{
    'Age_at_enrollment': age,
    'Marital_status': marital_status,
    'Course': course,
    'Nacionality': nacionality,
    'Displaced': binary_map[displaced],
    'Educational_special_needs': binary_map[educational_special_needs],
    'Debtor': binary_map[debtor],
    'Tuition_fees_up_to_date': binary_map[tuition_up_to_date],
    'Gender': binary_map[gender],
    'Scholarship_holder': binary_map[scholarship_holder],
    'International': binary_map[international]
}])

# Prediksi saat tombol diklik
if st.button("Prediksi Status"):
    # Preprocess data
    X_encoded = preprocessor.transform(input_data)
    
    # Prediksi
    prediction = model.predict(X_encoded)
    pred_label = label_encoder.inverse_transform(prediction)[0]
    
    # Tampilkan hasil
    st.subheader(f"Prediksi Status Mahasiswa: **{pred_label}**")
