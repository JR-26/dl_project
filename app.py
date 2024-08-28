import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the model and the new scaler fitted on 7 features
model = load_model('hcv_model.h5')
scaler = joblib.load('scaler_7_features.pkl')

# Create Streamlit app
st.title('Disease Classification Model')

st.write("Enter the following information:")

# Define input fields for each of the 7 selected features
epigastric_pain = st.number_input('Epigastric pain')
plat = st.number_input('Plat')
rna4 = st.number_input('RNA 4')
alt48 = st.number_input('ALT 48')
alt12 = st.number_input('ALT 12')
alt1 = st.number_input('ALT 1')
bhg = st.number_input('Baseline histological Grading')

# Create a button for prediction
if st.button('Predict'):
    # Prepare input data for prediction using only the 7 selected features
    input_data = np.array([[epigastric_pain, plat, rna4, alt48, alt12, alt1, bhg]])
    
    # Standardize the input data using the new scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the model
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Display the result
    st.write(f"Predicted class: {predicted_class[0] + 1}")  # Add 1 to adjust to original class labels (1-4)
