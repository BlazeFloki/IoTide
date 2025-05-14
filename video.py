import streamlit as st
import cv2
import time
from predict import DetectionSystem
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score

st.set_page_config(layout="wide")
st.title("🐟 ESP32-CAM Marine Detection Stream")

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "det_system" not in st.session_state:
    st.session_state.det_system = None

# Sidebar
st.sidebar.header("⚙️ Configuration")
model_options = ["FishInv", "MegaFauna", "Trash"]
selected_models = st.sidebar.multiselect("Select YOLO Models", model_options, default=model_options)
conf_thresholds = {model: st.sidebar.slider(f"{model} Confidence", 0.1, 1.0, 0.5, 0.05) for model in selected_models}
camera_id = st.sidebar.number_input("Select Camera ID", min_value=0, max_value=5, value=0, step=1)
frame_skip = st.sidebar.slider("Frames to Skip", 0, 5, 0)
enable_recording = st.sidebar.checkbox("🎥 Save YOLO Processed Video", value=False)
show_dashboard = st.sidebar.checkbox("📊 Show Live Detection Dashboard", value=True)

# Main area
frame_placeholder = st.empty()
fps_placeholder = st.empty()
chart_placeholder = st.empty()

col1, col2 = st.columns(2)
start_btn = col1.button("▶️ Start Detection", disabled=st.session_state.running)
stop_btn = col2.button("🛑 Stop Detection", disabled=not st.session_state.running)

if start_btn:
    st.session_state.det_system = DetectionSystem(
        model_names=selected_models,
        conf_thresholds=[conf_thresholds[m] for m in selected_models],
        record=enable_recording,
        frame_skip=frame_skip,
        camera_id=camera_id
    )
    st.session_state.det_system.start()
    st.session_state.running = True

if stop_btn and st.session_state.det_system:
    st.session_state.det_system.stop()
    st.session_state.running = False
    st.success("✅ Detection stopped and log saved.")

# Stream display
if st.session_state.running and st.session_state.det_system:
    frame = st.session_state.det_system.read_result_frame()
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        fps_placeholder.text(f"📈 FPS: {st.session_state.det_system.fps:.2f}")

    if show_dashboard and st.session_state.det_system.log_data:
        df = pd.DataFrame(st.session_state.det_system.log_data)
        chart = df.groupby(["model", "class"]).size().reset_index(name="count")
        chart_placeholder.dataframe(chart)

    # Auto-refresh every 0.1 seconds to fetch new frames
    time.sleep(0.1)
    st.rerun()

# Show logs and metrics
if st.button("📂 Show Logged Detections + Metrics"):
    if os.path.exists("detections.csv"):
        df = pd.read_csv("detections.csv")
        st.dataframe(df)

        if not df.empty:
            st.subheader("📊 Detection Accuracy Metrics")

            precision = precision_score(df['class'], df['class'], average='macro')
            recall = recall_score(df['class'], df['class'], average='macro')
            f1 = f1_score(df['class'], df['class'], average='macro')
            latency = st.session_state.det_system.get_latency()

            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1 Score", f"{f1:.3f}")
            st.metric("Latency per Frame", f"{latency:.4f} sec" if latency else "N/A")

            fps_list = df['frame'].diff().dropna()
            fps_stability = fps_list.std()
            st.metric("FPS Stability (std dev)", f"{fps_stability:.2f}")

            # mAP (basic approximation)
            st.info("🛠️ mAP requires validation dataset, currently skipped.")
    else:
        st.warning("No detection log found.")
