import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from product_db.match_engine import SkincareMatchingEngine

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
st.title("Skincare Analysis & Product Matching with SkinGPT")
st.write("Select your budget and upload a photo to analyze your skin condition below.")
st.markdown("---")

# Load model once
model = load_model()
classes = ['Acne', 'Blackheads', 'Dark Spots', 'Pores', 'Wrinkles']  # Class order must match training labels

# Budget selector moved above the uploader so users can set it before analysis
price_tier = st.selectbox("Select Your Budget", ["Budget", "Mid-Range", "Premium", "Luxury"])

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
                logits = model(input_tensor)[0]
                output_vector = logits.tolist()
                display_scores = torch.sigmoid(logits).mul(100).tolist()
            
            # C. Output Vector Results
            st.markdown("---")
            st.subheader("Skin Condition Scores")
            st.caption("Displayed scores are normalized for readability.")
            
            # Display scores in columns for cleaner presentation
            cols = st.columns(len(classes))
            for col, condition, score in zip(cols, classes, display_scores):
                with col:
                    st.metric(condition, f"{score:.1f}%")
                    st.progress(int(round(max(0.0, min(100.0, score)))))
            
            # D. PRODUCT MATCHING ALGORITHM
            st.markdown("---")
            st.subheader("Personalized Skincare Recommendations")
            
            # Append 0 for eyebags (not detected by model)
            user_vector = output_vector + [0]
            
            # Check if product database exists
            db_path = Path("dataset/final_sephora_database.csv")
            if not db_path.exists():
                st.error(f"❌ Product database not found at {db_path}. Please ensure the CSV is present or upload it.")
            else:
                try:
                    # Initialize the matching engine
                    engine = SkincareMatchingEngine(str(db_path))
                    
                    # Run the matching algorithm using the budget selected above
                    routine = engine.build_routine(user_vector, price_tier)
                    
                    # Display Results
                    if isinstance(routine, str):
                        # Error message returned by the engine
                        st.warning(routine)
                    else:
                        # Render each routine step as an inline card (no dropdowns)
                        for step, product in routine.items():
                            st.write(f"### {step}")
                            if isinstance(product, dict):
                                # Row layout: stack fields vertically for readability
                                product_name = product.get('product_name', 'Unknown')
                                brand_name = product.get('brand_name', 'Unknown')
                                price = product.get('price_usd', 0.0)
                                rating = product.get('rating', 0.0)
                                match_pct = product.get('match_percent', 0.0)
                                actives_list = product.get('clean_ingredient_array', []) or []

                                # Name and brand (prominent)
                                st.markdown(f"<div style='font-size:20px; font-weight:700'>{product_name}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='font-size:15px; color:#444; margin-bottom:6px'>By {brand_name}</div>", unsafe_allow_html=True)

                                # Price / Rating / Match as a single row of small metrics
                                cols = st.columns([1,1,1])
                                cols[0].metric("Price", f"${price:.2f}")
                                cols[1].metric("Rating", f"{rating:.1f}/5.0")
                                cols[2].metric("Match", f"{match_pct:.1f}%")

                                # Active ingredients on its own row
                                actives = ', '.join(actives_list)
                                if len(actives) > 300:
                                    actives = actives[:300].rsplit(',', 1)[0] + '...'
                                st.write(f"**Active Ingredients:** {actives}")

                                # Separator between products
                                st.markdown("---")
                            else:
                                st.write(product)
                except Exception as e:
                    st.error(f"❌ Error running matching algorithm: {e}")