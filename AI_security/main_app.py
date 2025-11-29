import cv2
import os
import time
import numpy as np
import torch
import glob
import threading
import queue
from ultralytics import YOLO
from sort import Sort
from datetime import datetime

# --- CONFIGURATION ---
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
YOLO_MODEL = "yolov8n.pt"

AI_W, AI_H = 640, 360
DISP_W, DISP_H = 960, 540

# --- SETUP ---
print("--- PHASE 2: TRACKING & DISPATCHING ---")
gpu = torch.cuda.is_available()
device = 0 if gpu else 'cpu'
model = YOLO(YOLO_MODEL)
model.to(device)

gate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
door_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Shared Memory for Identities
face_results = {}

def capture_thread(url, queue_list, index):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            continue
        queue_list[index] = frame

# Dispatcher: Sends images to Face Worker
def dispatch_face_job(img, job_id):
    try:
        cv2.imwrite(f"jobs_face/{job_id}.jpg", img)
    except: pass

# Result Reader: Reads answers from Face Worker
def check_face_results():
    for f in glob.glob("results_face/*.txt"):
        try:
            job_id = os.path.basename(f).replace('result_', '').replace('.txt', '')
            with open(f, 'r') as file: 
                name = file.read().strip()
                face_results[job_id] = name
                print(f"[BRAIN] {job_id} identified as: {name}")
            os.remove(f)
        except: pass

# Start Capture
frames = [None, None]
threading.Thread(target=capture_thread, args=(RTSP_URL_GATE, frames, 0), daemon=True).start()
threading.Thread(target=capture_thread, args=(RTSP_URL_DOOR, frames, 1), daemon=True).start()

# Clean folders
for d in ["jobs_face", "results_face"]:
    if not os.path.exists(d): os.makedirs(d)
    for f in glob.glob(f"{d}/*"): os.remove(f)

print("[SYSTEM] Ready. Waiting for video...")
time.sleep(3)

while True:
    if frames[0] is None or frames[1] is None:
        time.sleep(0.1); continue

    # Resize for AI
    ai_frames = [cv2.resize(f, (AI_W, AI_H)) for f in frames]
    
    # Batch Inference
    results = model(ai_frames, verbose=False, classes=[0]) 

    display_frames = []
    check_face_results() # Check for names every loop
    
    for i, res in enumerate(results):
        cam_name = "GATE" if i == 0 else "DOOR"
        tracker = gate_tracker if i == 0 else door_tracker
        
        # Get Detections
        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            if conf > 0.4: dets.append([x1, y1, x2, y2, conf])
        
        # Update Tracker
        tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))
        
        # Draw
        disp = cv2.resize(frames[i], (DISP_W, DISP_H))
        sx = DISP_W / AI_W
        sy = DISP_H / AI_H

        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t)
            uid = f"{cam_name.lower()}_{tid}"
            
            # --- 1. CRAWLING LOGIC (From Phase 1) ---
            w, h = x2 - x1, y2 - y1
            aspect = w / h
            is_low = y2 > (AI_H * 0.9)
            
            status = "STANDING"
            color = (0, 255, 0)
            
            if aspect > 1.2 and is_low:
                status = "CRAWLING"
                color = (0, 0, 255)
                print(f"[ALERT] {uid} is CRAWLING!")

            # --- 2. FACE DISPATCH LOGIC (New for Phase 2) ---
            # Only send door camera images to Face Worker
            if cam_name == "DOOR":
                # Send crop to worker
                crop = ai_frames[i][max(0,y1):min(AI_H,y2), max(0,x1):min(AI_W,x2)]
                if crop.size > 0:
                    dispatch_face_job(crop, uid)
                
                # Get Identity from memory
                identity = face_results.get(uid, "Scanning...")
                
                # Color logic for identity
                if identity == "puneeth": color = (0, 255, 0) # Green
                elif identity == "Unknown": color = (0, 0, 255) # Red
                
                label = f"{uid} | {status} | {identity}"
                cv2.rectangle(disp, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), color, 2)
                cv2.putText(disp, label, (int(x1*sx), int(y1*sy)-10), 1, 1.5, color, 2)
            else:
                # Gate Camera Logic (Just Crawling for now)
                label = f"{uid} | {status}"
                cv2.rectangle(disp, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), color, 2)
                cv2.putText(disp, label, (int(x1*sx), int(y1*sy)-10), 1, 1.5, color, 2)

        display_frames.append(disp)

    combined = cv2.hconcat(display_frames)
    cv2.imshow("Phase 2 Test", combined)
    if cv2.waitKey(1) == ord('q'): break

cv2.destroyAllWindows()