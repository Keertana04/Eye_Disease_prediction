import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import os
from io import BytesIO
import base64

# Load trained model
MODEL_PATH = "mobilenetv10_model.h5"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# CSV File Path
CSV_FILE = "patient_data.csv"

# Function to get the next available ID
def get_next_patient_id():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if not df.empty:
            return df["ID"].max() + 1  # Get the max ID and increment
    return 1  # Start from 1 if no records exist

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions
    return img

# Function to encode image for CSV storage
def encode_image(image):
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    return base64.b64encode(img_bytes.getvalue()).decode()

# Function to save data to CSV
def save_to_csv(patient_list):
    existing_data = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
    patient_df = pd.DataFrame(patient_list)
    updated_data = pd.concat([existing_data, patient_df], ignore_index=True)
    updated_data.to_csv(CSV_FILE, index=False)

# Function to clear patient records
def clear_records():
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        st.success("✅ All patient records have been cleared!")

# Streamlit UI Configuration
st.set_page_config(page_title="MediScan", page_icon="👁", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 📌 Navigation")
    page = st.radio("Select an Option", ["🏠 Home", "🔍 Predict Disease", "📋 View Patient Records"])

# Home Page
if page == "🏠 Home":
    st.markdown('<h1 style="text-align: center;">🩺 MediScan: AI Medical Diagnosis</h1>', unsafe_allow_html=True)
    
    st.markdown("<h3>Revolutionizing Eye Care with AI</h3>", unsafe_allow_html=True)
    st.markdown("""
        MediScan is a state-of-the-art medical imaging platform designed to streamline diagnostics.
        It allows healthcare professionals to manage, analyze, and generate detailed reports for retinal images.
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <h3>Key Features:</h3>
        <ul>
            <li>✔ <b>Patient Management:</b> Add and organize patient details.</li>
            <li>✔ <b>Image Analysis:</b> Upload and process retinal scans for diagnostic purposes.</li>
            <li>✔ <b>Report Generation:</b> Create comprehensive and detailed medical reports.</li>
            <li>✔ <b>User-Friendly:</b> Developed for efficiency, precision, and ease of use.</li>
        </ul>
    """, unsafe_allow_html=True)  

# Prediction Page
elif page == "🔍 Predict Disease":
    st.markdown('<h1 style="text-align: center;">🔬 Eye Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>This application predicts common eye diseases based on the uploaded image.</p>"
        , unsafe_allow_html=True)
    num_patients = st.number_input("Select Number of Patients", min_value=1, max_value=5, step=1, value=1)
    patient_details = []
    
    next_id = get_next_patient_id()  # Get the next available ID from CSV

    with st.form("patient_form"):
        st.markdown("### Enter Patient Details")
        
        for i in range(num_patients):
            patient_number = i + 1  # Keeps numbering consistent (1, 2, 3...)
            st.markdown(f"#### 👤 Patient {patient_number}")
            col1, col2 = st.columns(2)
            with col1:
                patient_name = st.text_input(f"Patient {patient_number} Name", placeholder="Enter Name", key=f"name_{i}")
            with col2:
                patient_age = st.number_input(f"Patient {patient_number} Age", min_value=1, max_value=120, step=1, value=25, key=f"age_{i}")

            uploaded_file = st.file_uploader(f"Upload Eye Image for Patient {patient_number}", type=["jpg", "png", "jpeg"], key=f"file_{i}")

            patient_details.append({
                "ID": next_id + i,  # Unique ID for storage
                "Patient Number": patient_number,
                "Patient Name": patient_name,
                "Age": patient_age,
                "File": uploaded_file
            })

        submit_button = st.form_submit_button("Predict Disease")

    if submit_button:
        records_to_save = []
        for patient in patient_details:
            if patient["File"] is not None:
                image = load_img(patient["File"])
                img_array = preprocess_image(image)
                predictions = model.predict(img_array)
                predicted_disease = class_labels[np.argmax(predictions)]
                encoded_img = encode_image(image)

                records_to_save.append({
                    "ID": patient["ID"],
                    "Patient Name": patient["Patient Name"],
                    "Age": patient["Age"],
                    "Disease": predicted_disease,
                    "Image": encoded_img
                })
                
                # Display output
                st.markdown(f"### Prediction Result for **Patient {patient['Patient Number']}**")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption=f"Uploaded Image - Patient {patient['Patient Number']}", width=150)
                with col2:
                    st.markdown(f"""
                    <div style="background-color:#20B2AA;padding:15px;border-radius:15px;text-align:center; font-weight:bold;">
                    <h4 style="color:#000000;font-size:25px;"> Name: {patient['Patient Name']}</h4>
                    <h4 style="color:#000000;font-size:25px;"> Age: {patient['Age']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background-color:#F4D03F;padding:15px;border-radius:15px;text-align:center; font-weight:bold; font-size:30px; box-shadow: 2px 2px 10px #888888;">
                <h3 style="color:#282828; font-size:30px;"> Predicted Disease: {predicted_disease}</h3>
                </div>
                """, unsafe_allow_html=True)

        save_to_csv(records_to_save)

# Patient Records Page
elif page == "📋 View Patient Records":
    st.markdown('<h1>📋 Patient Records</h1>', unsafe_allow_html=True)
    
    if os.path.exists(CSV_FILE):
        patient_data = pd.read_csv(CSV_FILE)
        
        # Decode and display images
        def decode_image(encoded_str):
            return BytesIO(base64.b64decode(encoded_str))
        
        patient_data["Image"] = patient_data["Image"].apply(lambda x: f'<img src="data:image/png;base64,{x}" width="50"/>')
        
        st.write(patient_data.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        csv_download = patient_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Patient Data", csv_download, "patient_data.csv", "text/csv")

        if st.button("🗑️ Clear Patient Records"):
            clear_records()
    else:
        st.warning("⚠️ No patient records found!")
