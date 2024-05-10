import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import tempfile
import os
import shutil
import glob
from io import BytesIO
import base64
import cv2
import numpy as np
import time

model = YOLO('yolov8n.pt')

def delete_directory(path):
    """Utility function to delete the specified directory."""
    if os.path.exists(path):
        shutil.rmtree(path)

def main():
    st.title('Object Detection with YOLOv8')

    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'], accept_multiple_files=False)

    if uploaded_file is not None:
        # Delete the results directory each time a new image is uploaded
        delete_directory('runs/detect/predict/')
        
        if uploaded_file.type.startswith('image'):
            process_image(uploaded_file)
        elif uploaded_file.type.startswith('video'):
            process_video(uploaded_file)

def process_image(uploaded_image):
    st.subheader("Image Detection")

    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
        image.save(temp_image.name)

        # Run inference and check for the creation of the directory
        with st.spinner('Running inference on the image...'):
            model.predict(temp_image.name, save=True, imgsz=320, conf=0.25)

        # Ensure the directory is recreated and check for the latest file
        if os.path.exists('runs/detect/predict/'):
            list_of_files = glob.glob('runs/detect/predict/*')
            latest_file = max(list_of_files, key=os.path.getctime) if list_of_files else None

            if latest_file:
                result_image = Image.open(latest_file)
                st.subheader("Detected Objects")
                st.image(result_image, caption='Result Image', use_column_width=True)

                # Add a download button for the result image
                if st.button("Download Result Image"):
                    download_image(result_image, "Result_Image")

            else:
                st.write("No detectable objects in the image or results are not saved.")
        else:
            st.write("No output directory found, check model configuration.")

def process_video(uploaded_video):
    st.subheader("Video Detection")

    # Display the uploaded video
    video_bytes = uploaded_video.read()
    st.video(video_bytes, format='video/mp4', start_time=0)

    # Process video frames
    with st.spinner('Processing video for object detection...'):
        temp_frames_dir = tempfile.mkdtemp()
        vidcap = cv2.VideoCapture()
        vidcap.open(uploaded_video.name)
        success, image = vidcap.read()
        frame_count = 0
        while success:
            frame_count += 1
            cv2.imwrite(os.path.join(temp_frames_dir, f"frame_{frame_count:04d}.jpg"), image)
            success, image = vidcap.read()

        # Run inference on each frame
        for frame_file in sorted(glob.glob(os.path.join(temp_frames_dir, '*.jpg'))):
            model.predict(frame_file, save=True, imgsz=320, conf=0.25)

        # Delete existing result video file if it exists
        if os.path.exists('result.mp4'):
            os.remove('result.mp4')

        # Create result video
        result_video_path = 'result.mp4'
        rebuild_video_from_jpgs("runs/detect/predict/", result_video_path)

        #result_video_file = open(result_video_path, 'rb')
        result_mp4_exists = os.path.exists('result.mp4')

        if result_mp4_exists:
            # Display the resulting video
            st.subheader("Detected Objects (Result Video)")
            video_file = open(result_video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, format='video/mp4', start_time=0)
            
            # Delete the results directory each time a new image is uploaded
            delete_directory('runs/detect/predict/')

        # Clean up
        #shutil.rmtree(temp_frames_dir)

def rebuild_video_from_jpgs(jpg_folder, output_video_path, fps=30):
    # Get all JPEG files in the folder
    jpg_files = [file for file in os.listdir(jpg_folder) if file.endswith('.jpg')]

    # Sort files numerically
    jpg_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    # Initialize VideoWriter
    frame = cv2.imread(os.path.join(jpg_folder, jpg_files[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    for jpg_file in jpg_files:
        frame = cv2.imread(os.path.join(jpg_folder, jpg_file))
        out.write(frame)

    # Release VideoWriter
    out.release()

def download_image(img, filename):
    """Downloads the given image."""
    img.save(filename + ".jpg", "JPEG")
    st.success(f"Downloaded {filename}.jpg")

if __name__ == "__main__":
    main()
