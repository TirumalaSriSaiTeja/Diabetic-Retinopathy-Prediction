import streamlit as st
import pandas as pd
import pickle

# Load the saved model and scaler
@st.cache_resource
def load_model():
    with open('retinopathy_svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

model = load_model()
scaler = load_scaler()

st.title('Diabetic Retinopathy Prediction')
st.write('Enter the patient information to predict the likelihood of diabetic retinopathy.')

# Input fields for patient data
age = st.number_input('Age', min_value=1, max_value=120, value=60)
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=50.0, max_value=200.0, value=100.0)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=30.0, max_value=150.0, value=80.0)
cholesterol = st.number_input('Cholesterol', min_value=50.0, max_value=300.0, value=100.0)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'age': age,
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'cholesterol': cholesterol
}])

if st.button('Predict'):
    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)
    prediction_proba = model.predict_proba(scaled_input_data)[:, 1]

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error(f'The model predicts: Retinopathy (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'The model predicts: No Retinopathy (Probability: {prediction_proba[0]:.2f})')

    st.write('---')
    st.write('Disclaimer: This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice.')
