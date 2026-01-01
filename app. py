import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Bone Density Classifier", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦴 Bone Density Classification System")
st.markdown("**EfficientNet-B0 Model** | Classify X-ray images into bone density categories")

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn. Dropout(p=0.2),
        nn.Linear(model.classifier[1].in_features, 3)
    )
    model.load_state_dict(torch.load('/content/drive/MyDrive/best_efficientnet_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Define transforms
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

# Prediction function
def predict(image, model, device, transform):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class]. item()
    return predicted_class, confidence, probabilities[0]. cpu().numpy()

# Load model
try:
    model, device = load_model()
    transform = get_transforms()
    class_names = ['Normal Bones Full', 'Osteopenia', 'Osteoporosis']
    class_descriptions = {
        'Normal Bones Full':  'Normal bone density - Healthy bones',
        'Osteopenia': 'Low bone density - Increased fracture risk',
        'Osteoporosis': 'Very low bone density - High fracture risk'
    }
    model_loaded = True
except Exception as e: 
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Sidebar
with st.sidebar:
    st. header("📊 Model Information")
    st.write("""
    ### Model Details
    - **Architecture**: EfficientNet-B0
    - **Test Accuracy**: 87.61%
    - **Classes**: 3
    
    ### Class Performance
    - **Normal Bones**:  100% accuracy ⭐
    - **Osteopenia**:  80% F1-score
    - **Osteoporosis**: 74% F1-score
    """)
    
    st.divider()
    st.header("ℹ️ About")
    st.write("""
    This application uses deep learning to classify bone density from X-ray images.
    
    **Classes:**
    - Normal Bones Full
    - Osteopenia
    - Osteoporosis
    """)

# Main content
if model_loaded:
    st.header("📤 Upload X-Ray Image")
    
    uploaded_file = st.file_uploader("Choose an X-ray image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st. subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        # Make prediction
        predicted_class, confidence, all_probs = predict(image, model, device, transform)
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            # Prediction card
            if predicted_class == 0:
                st.success(f"**{class_names[predicted_class]}**")
            elif predicted_class == 1:
                st.warning(f"**{class_names[predicted_class]}**")
            else:
                st.error(f"**{class_names[predicted_class]}**")
            
            st.write(f"*{class_descriptions[class_names[predicted_class]]}*")
            
            # Confidence metric
            st.metric("Confidence Score", f"{confidence*100:.2f}%", 
                     delta=None if confidence > 0.7 else "⚠️ Low confidence")
            
            st.divider()
            
            # Class probabilities
            st.subheader("📈 Class Probabilities")
            prob_data = {}
            for i, (class_name, prob) in enumerate(zip(class_names, all_probs)):
                prob_data[class_name] = prob
                col_left, col_right = st.columns([3, 1])
                with col_left: 
                    st. progress(prob)
                with col_right: 
                    st.write(f"{prob*100:.1f}%")
        
        # Additional visualization
        st.divider()
        st.subheader("📊 Confidence Distribution")
        
        fig, ax = plt. subplots(figsize=(10, 4))
        colors = ['green', 'orange', 'red']
        bars = ax.barh(class_names, all_probs, color=colors, alpha=0.7)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.set_title('Model Confidence by Class')
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, all_probs)):
            ax.text(prob + 0.02, i, f'{prob:. 2%}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    else:
        st.info("👆 Please upload an X-ray image to get started")
        
        # Example info
        st. markdown("""
        ### How to use:
        1. Upload an X-ray image (JPG or PNG format)
        2. The model will analyze the image
        3. View the prediction and confidence score
        4. Check class probabilities
        
        ### Expected Input: 
        - Clear X-ray images of bones
        - Image size: 224×224 pixels (automatically resized)
        - Format: JPG or PNG
        """)

else:
    st.error("⚠️ Failed to load the model. Please check the model path.")
