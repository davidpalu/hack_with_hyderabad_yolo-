import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import sys
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Live Safety Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --- FUNCTION TO GET CORRECT FILE PATH ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO object detection model from the specified path."""
    model = YOLO(model_path)
    return model

# --- UI & APP LOGIC ---
st.title("🛡️ Live Safety Detection Dashboard")
st.write("This dashboard uses a live webcam feed to detect safety objects in real-time.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Model path
    model_path = resource_path("best.pt")
    
    # Confidence Threshold Slider
    confidence = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # Webcam Selection
    # You might need to change this index if you have multiple cameras
    webcam_index = 0 

# Load the model
try:
    model = load_model(model_path)
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

def start_detection():
    st.session_state.run_detection = True

def stop_detection():
    st.session_state.run_detection = False

with st.sidebar:
    st.button("Start Detection", on_click=start_detection, type="primary")
    st.button("Stop Detection", on_click=stop_detection)

# --- MAIN DISPLAY ---
frame_placeholder = st.empty()
stats_placeholder = st.empty()

if st.session_state.run_detection:
    try:
        vid_cap = cv2.VideoCapture(webcam_index)
        if not vid_cap.isOpened():
            st.error("Error: Could not open webcam. Please check connection and permissions.")
        else:
            st.sidebar.success("Webcam connected successfully!")
            
            prev_time = 0
            
            while vid_cap.isOpened() and st.session_state.run_detection:
                success, frame = vid_cap.read()
                if not success:
                    st.write("The video stream has ended or failed.")
                    break

                # --- Detection Logic ---
                results = model.predict(frame, conf=confidence)
                annotated_frame = results[0].plot()
                
                # --- Performance Metrics ---
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                # --- Display Video Frame ---
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, channels="RGB")

                # --- Display Stats ---
                detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]
                object_counts = {obj: detected_objects.count(obj) for obj in set(detected_objects)}

                stats_placeholder.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
                    <h3 style="color: #333;">Real-time Stats</h3>
                    <b>Performance:</b> {fps:.2f} FPS <br>
                    <b>Detections:</b> {len(detected_objects)}
                </div>
                """, unsafe_allow_html=True)
                
                if object_counts:
                    st.sidebar.subheader("Detected Objects")
                    for obj, count in object_counts.items():
                        st.sidebar.metric(label=obj, value=count)

            vid_cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        st.error(f"An error occurred during detection: {e}")

else:
    st.info("Click 'Start Detection' in the sidebar to begin the live feed.")