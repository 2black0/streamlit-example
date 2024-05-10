import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

model = YOLO('yolov8n.pt')

def main():
    st.title('Image Upload and Object Detection with YOLOv8')

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1.header("Original Image")
        col1.image(image, use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image.save(temp_image.name)

            results = model.predict(temp_image.name, save=True, imgsz=320, conf=0.25)  # Lower conf for testing
            if results:
                if hasattr(results, 'files') and results.files:  # Check if files exist
                    result_image = Image.open(results.files[0])
                    col2.header("Detected Objects")
                    col2.image(result_image, use_column_width=True)
                else:
                    col2.write("Results were generated but no files were saved.")
            else:
                col2.write("No detectable objects in the image or model did not generate results.")

    if st.button("Clear Image"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()