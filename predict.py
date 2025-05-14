import cv2
import time
import threading
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os

MODEL_PATHS = {
    "FishInv": r"C:\Users\Soham\OneDrive\Desktop\MIT Assignments\SEM 6\Mini Project\marine-detect\src\marine_detect\FishInv.pt",
    "MegaFauna": r"C:\Users\Soham\OneDrive\Desktop\MIT Assignments\SEM 6\Mini Project\marine-detect\src\marine_detect\MegaFauna.pt",
    "Trash": r"C:\Users\Soham\OneDrive\Desktop\MIT Assignments\SEM 6\Mini Project\marine-detect\src\marine_detect\Trash.pt"
}
OUTPUT_VIDEO_PATH = r"E:\mpr\outputvideo.avi"
LOG_PATH = "detections.csv"

class DetectionSystem:
    def __init__(self, model_names, conf_thresholds, record=False, frame_skip=0, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.lock = threading.Lock()
        self.running = False
        self.frame = None
        self.result_frame = None
        self.models = [YOLO(MODEL_PATHS[m]) for m in model_names]
        self.conf_thresholds = conf_thresholds
        self.frame_count = 0
        self.record = record
        self.fps = 0
        self.video_writer = None
        self.log_data = []
        self.start_time = None
        self.total_processed_frames = 0
        self.frame_skip = frame_skip  # ⬅️ added frame skipping
        self.skip_counter = 0

    def start(self):
        self.running = True
        self.start_time = time.time()
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.detection_loop, daemon=True).start()

    def stop(self):
        self.running = False
        time.sleep(1)
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        if self.log_data:
            self.write_log()

    def capture_loop(self):
        prev_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            with self.lock:
                self.frame = frame.copy()
            self.frame_count += 1

            now = time.time()
            self.fps = 1 / (now - prev_time)
            prev_time = now

            if self.record:
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 10, (224, 224))
                self.video_writer.write(frame)

            time.sleep(0.01)  # ~100 fps capture

    def detection_loop(self):
        while self.running:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None
            if frame is None:
                continue

            # Frame skipping logic
            if self.skip_counter < self.frame_skip:
                self.skip_counter += 1
                continue
            else:
                self.skip_counter = 0

            result_frame = frame.copy()
            for i, model in enumerate(self.models):
                results = model(result_frame, conf=self.conf_thresholds[i])
                for r in results:
                    result_frame = r.plot(img=result_frame)
                    boxes = r.boxes
                    if boxes is not None and len(boxes.cls) > 0:
                        for cls, conf in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                            log_row = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "frame": self.frame_count,
                                "model": list(MODEL_PATHS.keys())[i],
                                "class": int(cls),
                                "confidence": round(float(conf), 3)
                            }
                            self.log_data.append(log_row)

            self.total_processed_frames += 1

            # Add FPS display
            cv2.putText(result_frame, f"FPS: {self.fps:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            with self.lock:
                self.result_frame = result_frame.copy()

    def read_result_frame(self):
        with self.lock:
            return self.result_frame.copy() if self.result_frame is not None else None

    def write_log(self):
        df = pd.DataFrame(self.log_data)
        header = not os.path.exists(LOG_PATH)
        df.to_csv(LOG_PATH, mode='a', header=header, index=False)

    def get_latency(self):
        if self.start_time and self.total_processed_frames > 0:
            total_time = time.time() - self.start_time
            return total_time / self.total_processed_frames
        return None
