import streamlit as st
from PIL import Image
from ultralytics import YOLO  # This should be adjusted based on actual import if yolov8 is available differently
from io import BytesIO
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
            temp_image.close()  # Close the file so it can be reopened by the model

            # Process the image using the model
            results = model([temp_image.name])
            results = results[0]

            # Display the object detection results
            col2.header("Detected Objects")
            result_image = Image.open(BytesIO(results.show(save=False)))
            col2.image(result_image, use_column_width=True)

    if st.button("Clear Image"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
