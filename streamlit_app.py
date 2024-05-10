import streamlit as st
from PIL import Image
from ultralytics import YOLO  # Make sure this import matches the package structure
import tempfile

# Load the YOLO model from local .pt file
model = YOLO('yolov8n.pt')

def main():
    st.title('Image Upload and Object Detection with YOLOv8')

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')  # Convert image to RGB
        col1.header("Original Image")
        col1.image(image, use_column_width=True)

        # Use a temporary file to handle the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image.save(temp_image.name)  # Save the image to a temporary file

            # Run inference on the image with specific arguments
            results = model.predict(temp_image.name, save=True, imgsz=320, conf=0.5)

            # Check for the saved output and display
            if results is not None and hasattr(results, 'files'):
                result_image = Image.open(results.files[0])  # Load the saved result image
                col2.header("Detected Objects")
                col2.image(result_image, use_column_width=True)
            else:
                col2.write("No detectable objects in the image or results are not saved.")

    if st.button("Clear Image"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
