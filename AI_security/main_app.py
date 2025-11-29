import cv2
import os
import time
import threading
import queue
import numpy as np
import glob
import torch
from ultralytics import YOLO
from sort import Sort
from datetime import datetime

# --- CONFIGURATION ---
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
YOLO_MODEL = "yolov8n.pt"

AI_W, AI_H = 640, 360
DISP_W, DISP_H = 960, 540

# --- INTELLIGENCE CONSTANTS ---
CRAWL_RATIO = 1.6
RUN_GROWTH = 1.40
RUN_SPEED_PIXELS = 50.0
WALK_SPEED_PIXELS = 10.0
FLOOR_LIMIT = 0.85

print("--- PHASE 2: INTEGRATED BRAIN (FIXED) ---")
gpu = torch.cuda.is_available()
device = 0 if gpu else 'cpu'

model = YOLO(YOLO_MODEL)
model.to(device)

gate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
door_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Master Memory: { id: { 'last_box': [], 'face': '...', 'pose': '...', 'status': 'IDLE' } }
entity_memory = {}

# --- WORKER FUNCTIONS ---
def dispatch_job(env, img, jid):
    try:
        folder = "jobs_face" if env == "face" else "jobs_pose"
        cv2.imwrite(f"{folder}/{jid}.jpg", img)
    except: pass

def check_worker_results():
    # Face Results
    for f in glob.glob("results_face/*.txt"):
        try:
            jid = os.path.basename(f).replace('result_', '').replace('.txt', '')
            with open(f, 'r') as file: res = file.read().strip()
            if jid in entity_memory:
                entity_memory[jid]['face'] = res
                if res != "Unknown" and res != "Verifying...":
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [IDENTITY] {jid} is {res}")
            os.remove(f)
        except: pass

    # Pose Results
    for f in glob.glob("results_pose/*.txt"):
        try:
            jid = os.path.basename(f).replace('result_', '').replace('.txt', '')
            with open(f, 'r') as file: res = file.read().strip()
            if jid in entity_memory:
                entity_memory[jid]['pose'] = res
                if "VIOLENCE" in res or "SURRENDER" in res:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [VIOLENCE] {jid}: {res}")
            os.remove(f)
        except: pass

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

# Start Threads
frames = [None, None]
threading.Thread(target=capture_thread, args=(RTSP_URL_GATE, frames, 0), daemon=True).start()
threading.Thread(target=capture_thread, args=(RTSP_URL_DOOR, frames, 1), daemon=True).start()

# Clean folders
for d in ["jobs_face", "results_face", "jobs_pose", "results_pose"]:
    if not os.path.exists(d): os.makedirs(d)
    for f in glob.glob(f"{d}/*"): os.remove(f)

print("[SYSTEM] Ready. Waiting for video...")
time.sleep(3)

while True:
    if frames[0] is None or frames[1] is None:
        time.sleep(0.1); continue

    ai_frames = [cv2.resize(f, (AI_W, AI_H)) for f in frames]
    results = model(ai_frames, verbose=False, classes=[0]) 

    display_frames = []
    
    # Check for updates from workers
    check_worker_results()

    for i, res in enumerate(results):
        cam_name = "GATE" if i == 0 else "DOOR"
        tracker = gate_tracker if i == 0 else door_tracker
        
        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if float(box.conf[0]) > 0.4: dets.append([x1, y1, x2, y2, float(box.conf[0])])
        
        tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))
        
        disp = cv2.resize(frames[i], (DISP_W, DISP_H))
        sx = DISP_W / AI_W
        sy = DISP_H / AI_H

        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t)
            uid = f"{cam_name}_{tid}"
            
            # Initialize Memory
            if uid not in entity_memory:
                entity_memory[uid] = {
                    'area': 0, 'center': (0,0), 'status': 'IDLE',
                    'face': 'Scanning...', 'pose': 'Normal'
                }
            
            mem = entity_memory[uid]
            
            # --- GEOMETRY LOGIC ---
            w, h = x2 - x1, y2 - y1
            aspect = w / h
            feet_y = y2
            
            # FIX: Calculate center BEFORE logic check
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if aspect > CRAWL_RATIO and feet_y > (AI_H * FLOOR_LIMIT):
                status = "CRAWLING"
            else:
                # Calculate Speed
                prev_cx, prev_cy = mem['center']
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                if dist > RUN_SPEED_PIXELS: status = "RUNNING"
                elif dist > WALK_SPEED_PIXELS: status = "WALKING"
                else: status = "IDLE"

            mem['status'] = status
            mem['area'] = w * h
            mem['center'] = (cx, cy)

            # --- DISPATCH TO WORKERS ---
            crop = ai_frames[i][max(0,y1):min(AI_H,y2), max(0,x1):min(AI_W,x2)]
            if crop.size > 0:
                if cam_name == "DOOR": 
                    dispatch_job("face", crop, uid)
                if cam_name == "GATE": 
                    dispatch_job("pose", crop, uid)

            # --- DRAWING ---
            if mem['face'] == 'puneeth': color = (0, 255, 0) # Green
            elif status == 'CRAWLING': color = (0, 0, 255) # Red
            elif status == 'RUNNING': color = (0, 0, 255) # Red
            elif "VIOLENCE" in mem['pose']: color = (0, 0, 255) # Red
            elif mem['face'] == 'Unknown': color = (0, 0, 255) # Red
            else: color = (0, 255, 255) # Yellow (Scanning)

            cv2.rectangle(disp, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), color, 2)
            
            # Label: ID | ACTION | IDENTITY
            label = f"{uid} | {status} | {mem['face']}"
            cv2.putText(disp, label, (int(x1*sx), int(y1*sy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        display_frames.append(disp)

    combined = cv2.hconcat(display_frames)
    cv2.imshow("Integrated Security", combined)
    if cv2.waitKey(1) == ord('q'): break

cv2.destroyAllWindows()