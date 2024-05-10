import streamlit as st
from PIL import Image
import torch
from io import BytesIO

# Function to load model with specified weights
def load_model():
    model_url = 'yolov8n.pt'
    model = torch.hub.load('ultralytics/yolov8', 'custom', path_or_model=model_url)
    return model

model = load_model()

def main():
    st.title('Image Upload and Object Detection with Custom YOLOv8 Model')
    
    # Create a file uploader for image files
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    
    # Container to hold the images
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        col1.header("Original Image")
        col1.image(image, use_column_width=True)
        
        # Convert image to format suitable for YOLO
        image_bytes = BytesIO(uploaded_file.getvalue())
        img = Image.open(image_bytes)
        results = model(img, size=640)
        
        # Display the object detection image
        col2.header("Detected Objects")
        col2.image(results.render()[0], use_column_width=True)

    # Clear button to remove the uploaded image
    if st.button("Clear Image"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()