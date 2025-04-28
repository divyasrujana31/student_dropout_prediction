# app.py

import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
with open('dropout_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title in sidebar
st.sidebar.title("🎯 Student Dropout Prediction")

st.sidebar.write("Fill out the student's details below:")

# Input fields in sidebar
G1 = st.sidebar.number_input('First Period Grade (G1)', min_value=0, max_value=20, value=10, help="First term grade (0-20)")
G2 = st.sidebar.number_input('Second Period Grade (G2)', min_value=0, max_value=20, value=10, help="Second term grade (0-20)")
studytime = st.sidebar.number_input('Weekly Study Time (hours)', min_value=1, max_value=50, value=5, help="Average study hours per week")
failures = st.sidebar.number_input('Number of Past Class Failures', min_value=0, max_value=5, value=0, help="Number of previous failed classes")
absences = st.sidebar.number_input('Number of Absences', min_value=0, max_value=100, value=5, help="Number of school absences")

# Main page
st.title("📚 Predict Student Dropout Risk")

st.write("Click the button below to make a prediction:")

# Predict button
if st.button('🔮 Predict'):
    # Prepare the input data
    input_data = np.array([[G1, G2, studytime, failures, absences]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Show confidence
    confidence = np.max(prediction_proba) * 100

    if prediction[0] == 1:
        st.error(f'⚠️ The student is at risk of dropping out. Confidence: {confidence:.2f}%')
    else:
        st.success(f'✅ The student is NOT at risk of dropping out. Confidence: {confidence:.2f}%')

    # Add Pie Chart Visualization
    st.subheader("📊 Prediction Probability Chart")

    labels = ['Not Dropout', 'Dropout']
    fig, ax = plt.subplots()
    ax.pie(prediction_proba[0], labels=labels, autopct='%1.1f%%', startangle=90, colors=['#00cc99', '#ff6666'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
