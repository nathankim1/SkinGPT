import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. ARCHITECTURE DEFINITION
def get_resnet18_model():
    model = models.resnet18(weights=None) # We will load our own weights
    num_ftrs = model.fc.in_features
    # Your project notes specify a 5-class output with Sigmoid for probabilities
    model.fc = nn.Linear(num_ftrs, 5)
    return model

# 2. CACHING THE MODEL
# Using st.cache_resource ensures the .pth file is only loaded into memory once
@st.cache_resource
def load_model():
    model = get_resnet18_model()
    # Ensure five_class_best.pth is in your project root directory
    model.load_state_dict(torch.load("five_class_best.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. PREPROCESSING PIPELINE
def preprocess_image(image):
    # Standard ResNet preprocessing: Resize, Crop, Tensor, Normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- STREAMLIT UI ---
st.title("Skincare Analysis: Model Integration")
st.write("Upload a photo to generate your Condition Vector.")

# Load model once
model = load_model()
classes = ['Acne', 'Eyebags', 'Hyperpigmentation', 'Wrinkles', 'Dryness']

uploaded_file = st.file_uploader("Upload Skin Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Target Image", width=300)
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):
            # A. Preprocess
            input_tensor = preprocess_image(img)
            
            # B. Feed the Model
            with torch.no_grad():
                output_vector = model(input_tensor)[0].tolist()
            
            # C. Output Vector Results
            st.subheader("Generated Condition Vector")
            st.json(dict(zip(classes, output_vector)))
            
            # Visualizing the vector for the team
            st.bar_chart(dict(zip(classes, output_vector)))