import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import shutil
import glob
from io import BytesIO
import base64

model = YOLO('yolov8n.pt')

def delete_directory(path):
    """Utility function to delete the specified directory."""
    if os.path.exists(path):
        shutil.rmtree(path)

def main():
    st.title('Image Upload and Object Detection with YOLOv8')

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    col1, col2 = st.columns([2, 2])

    if uploaded_file is not None:
        # Delete the results directory each time a new image is uploaded
        delete_directory('runs/detect/predict/')

        image = Image.open(uploaded_file).convert('RGB')
        col1.header("Original Image")
        col1.image(image, use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image.save(temp_image.name)

            # Run inference and check for the creation of the directory
            model.predict(temp_image.name, save=True, imgsz=320, conf=0.25)

            # Ensure the directory is recreated and check for the latest file
            if os.path.exists('runs/detect/predict/'):
                list_of_files = glob.glob('runs/detect/predict/*')
                latest_file = max(list_of_files, key=os.path.getctime) if list_of_files else None

                if latest_file:
                    result_image = Image.open(latest_file)
                    col2.header("Detected Objects")
                    col2.image(result_image, use_column_width=True)

                    # Add a download button for the result image
                    if st.button("Download Result Image"):
                        download_image(result_image, "Result_Image")

                else:
                    col2.write("No detectable objects in the image or results are not saved.")
            else:
                col2.write("No output directory found, check model configuration.")

def download_image(img, filename):
    """Downloads the given image."""
    img.save(filename + ".jpg", "JPEG")
    st.success(f"Downloaded {filename}.jpg")

if __name__ == "__main__":
    main()
