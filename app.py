import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import os

# Load trained MobileNetV3 model
MODEL_PATH = "mobilenetv10_model.h5"  # Ensure this file exists in the same directory
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['Cataract','Diabetic Retinopathy', 'Glaucoma', 'Normal']

# CSV File Path
CSV_FILE = "patient_data_3.csv"

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions
    return img

# Function to save data to CSV
def save_to_csv(patient_name, age, disease):
    new_entry = pd.DataFrame({
        "ID": [len(pd.read_csv(CSV_FILE)) + 1 if os.path.exists(CSV_FILE) else 1],
        "Patient Name": [patient_name],
        "Age": [age],
        "Disease": [disease]
    })
    
    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
    else:
        updated_data = new_entry
        
    updated_data.to_csv(CSV_FILE, index=False)
    return updated_data

# Function to clear patient records
def clear_records():
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    st.success("✅ Patient records have been cleared!")

# Streamlit UI
st.set_page_config(page_title="Eye Disease Detection", page_icon="👁", layout="wide")

# Sidebar Navigation
st.sidebar.markdown("### 📌 Navigation")
page = st.sidebar.radio("Select an Option", ["🔍 Predict Disease", "📋 View Patient Records"])

# Prediction Page
if page == "🔍 Predict Disease":
    st.markdown("""
        <h1 style='text-align: center; color: #1A5276;'>🩺 MediScan - Eye Disease Detection</h1>
        <p style='text-align: center; font-size:18px; color:#154360;'>This application predicts common eye diseases based on the uploaded image.</p>
        """, unsafe_allow_html=True)

    # Collect User Information
    with st.form("patient_form"):
        st.markdown("### Enter Patient Details")
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="Enter your Name")
        with col2:
            patient_age = st.number_input("Age", min_value=1, max_value=120, step=1, value=1)

        uploaded_file = st.file_uploader("Upload an Eye Image", type=["jpg", "png", "jpeg"])

        submit_button = st.form_submit_button("Predict Disease")

    # Process Prediction
    if submit_button and uploaded_file is not None:
        image = load_img(uploaded_file)
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_disease = class_labels[np.argmax(predictions)]

        # Save Data
        updated_data = save_to_csv(patient_name, patient_age, predicted_disease)

        # Display Uploaded Image & Patient Details
        st.markdown("### Patient Details & Image")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", width=150)
        with col2:
            st.markdown(f"""
                <div style="background-color:#20B2AA;padding:15px;border-radius:20px;text-align:center; font-weight:bold;">
                <h4 style="color:#000000;font-size:30px;"> Name: {patient_name}</h4>
                <h4 style="color:#000000;font-size:30px;"> Age: {patient_age}</h4>
                </div>
                """, unsafe_allow_html=True)

        # Display Prediction
        st.markdown("### Prediction Result")
        st.markdown(f"""
        <div style="background-color:#F4D03F;padding:15px;border-radius:15px;text-align:center; font-weight:bold; font-size:36px; box-shadow: 2px 2px 10px #888888;">
        <h3 style="color:#282828; font-size:35px;"> Predicted Disease: {predicted_disease}</h3>
        </div>
        """, unsafe_allow_html=True)

# Patient Records Page
elif page == "📋 View Patient Records":
    st.markdown("### 📋 Patient Records")

    if os.path.exists(CSV_FILE):
        patient_data = pd.read_csv(CSV_FILE)
        st.dataframe(patient_data)

        # Provide Download Option
        csv_download = patient_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Patient Data", csv_download, "patient_data.csv", "text/csv")

        # Clear Records Button
        if st.button("🗑️ Clear Patient Records"):
            clear_records()
    else:
        st.warning("⚠️ No patient records found!")
