import streamlit as st
from PIL import Image
import time

st.set_page_config(page_title="Skincare Analysis AI", layout="centered")

st.title("✨ Skincare Analysis App")
st.write("Upload a photo for an instant skin health report.")

# --- SIDEBAR (Infrastructure/Settings) ---
st.sidebar.header("Settings")
model_version = st.sidebar.selectbox("Model Version", ["v1.0-beta", "v1.1-latest"])

# --- IMAGE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Skin'):
        with st.spinner('Running deep learning model...'):
            # This is where your team's model code will go
            time.sleep(2) # Simulating processing time
            st.success("Analysis Complete!")
            
            # Placeholder for results
            col1, col2 = st.columns(2)
            col1.metric("Hydration Level", "72%", "+5%")
            col2.metric("Sensitivity Score", "Low", "Stable")