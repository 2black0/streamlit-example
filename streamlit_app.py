import streamlit as st
from PIL import Image
from ultralytics import YOLO
from io import BytesIO

# Load the YOLO model from local .pt file
model = YOLO('yolov8n.pt')

def main():
    st.title('Image Upload and Object Detection with YOLOv8')

    # Create a file uploader for image files
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    
    # Container to hold the images
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        col1.header("Original Image")
        col1.image(image, use_column_width=True)

        # Convert image to a format suitable for YOLO
        image_bytes = BytesIO(uploaded_file.getvalue())
        image.save("temp_image.jpg")  # Save the image to a temporary file to be processed by YOLO

        # Run object detection
        results = model(['temp_image.jpg'])  # Process the image using the model
        results = results[0]  # Get results for the first image (we only process one image here)
        
        # Display the object detection results
        col2.header("Detected Objects")
        result_image = Image.open(results.show(save=False))  # Display detection result without saving
        col2.image(result_image, use_column_width=True)

    # Clear button to remove the uploaded image and results
    if st.button("Clear Image"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
