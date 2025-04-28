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
    # Prepare the input data (ensure it is a 2D array)
    input_data = np.array([[G1, G2, studytime, failures, absences]])

    # Check for missing values
    if np.any(np.isnan(input_data)):
        st.error("Please fill out all fields. No missing values allowed.")
    else:
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Show confidence
        confidence = np.max(prediction_proba) * 100

        # Display prediction result with confidence
        if prediction[0] == 1:
            st.error(f'⚠️ The student is at risk of dropping out. Confidence: {confidence:.2f}%')
        else:
            st.success(f'✅ The student is NOT at risk of dropping out. Confidence: {confidence:.2f}%')

        # Pie chart for prediction probabilities
        labels = ['Not Dropout', 'Dropout']
        fig, ax = plt.subplots()
        ax.pie(prediction_proba[0], labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is a circle.

        # Display pie chart in Streamlit
        st.pyplot(fig)

