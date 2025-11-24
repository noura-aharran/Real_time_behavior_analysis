# src/app/app.py
import streamlit as st
import cv2
import numpy as np
import time
import os
import sys
# make sure project root is in path to import src.core
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.core.capture import VideoCaptureAsync
from src.core.detector import Detector

st.set_page_config(layout="wide", page_title="RT Behavior MVP")

st.title("RT Behavior — MVP")
st.write("Prototype: capture → detection personne → affichage")

# Sidebar
source = st.sidebar.text_input("Source (0 for webcam or rtsp url)", "0")
use_yolo = st.sidebar.checkbox("Use YOLOv8 if available", True)
conf_thres = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.35)
model_path = st.sidebar.text_input("YOLO model path (optional)", "")

start = st.sidebar.button("Start")
stop = st.sidebar.button("Stop")

placeholder = st.empty()
status_text = st.empty()

# keep objects in session_state so they persist across reruns
if "cap" not in st.session_state:
    st.session_state.cap = None
if "detector" not in st.session_state:
    st.session_state.detector = None
if "running" not in st.session_state:
    st.session_state.running = False

if start and not st.session_state.running:
    st.session_state.cap = VideoCaptureAsync(source)
    st.session_state.cap.start()
    if use_yolo:
        det = Detector(model_path=model_path if model_path else None, conf_thres=conf_thres)
    else:
        det = Detector(model_path=None, conf_thres=conf_thres)
    st.session_state.detector = det
    st.session_state.running = True

if stop and st.session_state.running:
    if st.session_state.cap:
        st.session_state.cap.stop()
    st.session_state.running = False

# Main loop
if st.session_state.running and st.session_state.cap:
    detector = st.session_state.detector
    cap = st.session_state.cap
    fps_smooth = 0.0
    last = time.time()
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            # show waiting
            status_text.info("Waiting for frames...")
            time.sleep(0.1)
            continue
        # detect
        dets = detector.detect(frame)
        # draw detections
        for d in dets:
            x1,y1,x2,y2 = d.to_xyxy()
            label = f"{d.cls} {d.conf:.2f}"
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(10,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # show FPS
        now = time.time()
        fps = 1.0 / (now - last) if now != last else 0.0
        fps_smooth = 0.9*fps_smooth + 0.1*fps
        last = now
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        # convert to RGB and display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(img, channels="RGB", use_column_width=True)
        # allow Streamlit to rerun widgets
        if st.sidebar.button("Stop"):
            st.session_state.cap.stop()
            st.session_state.running = False
            break
else:
    st.info("App stopped. Click Start to run the pipeline.")
