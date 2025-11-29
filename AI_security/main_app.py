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

# Resolutions
AI_W, AI_H = 640, 360
DISP_W, DISP_H = 960, 540

# --- INTELLIGENCE THRESHOLDS ---
CRAWL_RATIO = 1.6 
FLOOR_LIMIT = 0.85 
RUN_GROWTH = 1.30 
RUN_SPEED_PIXELS = 50.0
WALK_SPEED_PIXELS = 10.0

# --- WEAPONS ---
WEAPON_CLASSES = [43, 76, 34] # Knife, Scissors, Bat
WEAPON_CONF = 0.20

print("--- PHASE 2: INTEGRATED BRAIN (VERBOSE) ---")
gpu = torch.cuda.is_available()
device = 0 if gpu else 'cpu'
print(f"[SYSTEM] Hardware: {device}")

model = YOLO(YOLO_MODEL)
model.to(device)

gate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
door_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

q_gate = queue.Queue(maxsize=2)
q_door = queue.Queue(maxsize=2)

# Memory: { id: { 'face': '...', 'pose': '...', 'status': 'IDLE', 'last_box': [], 'center':() } }
entity_memory = {}

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

def dispatch(env, img, jid):
    try: 
        folder = "jobs_face" if env == "face" else "jobs_pose"
        # Only save if file doesn't exist (don't spam workers)
        path = f"{folder}/{jid}.jpg"
        if not os.path.exists(path):
            cv2.imwrite(path, img)
            # print(f"[DISPATCH] Sent {jid} to {env}")
    except: pass

def check_results():
    # Face Results
    for f in glob.glob("results_face/*.txt"):
        try:
            jid = os.path.basename(f).split('.')[0].replace('result_','')
            with open(f, 'r') as file: res = file.read().strip()
            
            if jid in entity_memory:
                old = entity_memory[jid].get('face')
                if old != res:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [IDENTITY UPDATE] {jid}: {res}")
                entity_memory[jid]['face'] = res
            os.remove(f)
        except: pass

    # Pose Results
    for f in glob.glob("results_pose/*.txt"):
        try:
            jid = os.path.basename(f).split('.')[0].replace('result_','')
            with open(f, 'r') as file: res = file.read().strip()
            
            if jid in entity_memory:
                entity_memory[jid]['pose'] = res
                if "VIOLENCE" in res or "SURRENDER" in res:
                     print(f"[{datetime.now().strftime('%H:%M:%S')}] [VIOLENCE REPORT] {jid}: {res}")
            os.remove(f)
        except: pass

# Init Folders
for d in ["jobs_face", "results_face", "jobs_pose", "results_pose"]:
    if not os.path.exists(d): os.makedirs(d)
    for f in glob.glob(f"{d}/*"): os.remove(f)

# Start Threads
frames = [None, None]
import threading
threading.Thread(target=capture_thread, args=(RTSP_URL_GATE, frames, 0), daemon=True).start()
threading.Thread(target=capture_thread, args=(RTSP_URL_DOOR, frames, 1), daemon=True).start()

print("[SYSTEM] Waiting for video...")
time.sleep(3)

frame_cnt = 0
active_weapons = []

while True:
    if frames[0] is None or frames[1] is None:
        time.sleep(0.1); continue

    frame_cnt += 1
    ai_frames = [cv2.resize(f, (AI_W, AI_H)) for f in frames]
    
    # Batch Inference (People + Weapons)
    results = model(ai_frames, verbose=False, classes=[0] + WEAPON_CLASSES, conf=WEAPON_CONF) 

    display_frames = []
    check_results() # Read worker files
    
    active_weapons = []

    for i, res in enumerate(results):
        cam_name = "GATE" if i == 0 else "DOOR"
        tracker = gate_tracker if i == 0 else door_tracker
        
        people_dets = []
        
        # Separate People from Weapons
        for box in res.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            if cls == 0 and conf > 0.4: # Person
                people_dets.append([x1, y1, x2, y2, conf])
            elif cls in WEAPON_CLASSES: # Weapon
                w_name = model.names[cls]
                active_weapons.append(w_name)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ALARM] WEAPON FOUND: {w_name}")
        
        tracks = tracker.update(np.array(people_dets) if people_dets else np.empty((0, 5)))
        
        disp = cv2.resize(frames[i], (DISP_W, DISP_H))
        sx = DISP_W / AI_W
        sy = DISP_H / AI_H

        # Draw Floor Line (Blue)
        floor_y_disp = int(DISP_H * FLOOR_LIMIT)
        cv2.line(disp, (0, floor_y_disp), (DISP_W, floor_y_disp), (255, 0, 0), 2)

        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t)
            uid = f"{cam_name}_{tid}"
            
            # --- LOGIC START ---
            w = x2 - x1
            h = y2 - y1
            cx, cy = (x1+x2)/2, (y1+y2)/2
            aspect = w / h
            feet_y = y2 
            floor_thresh = AI_H * FLOOR_LIMIT
            
            # 1. Posture Detection (Sitting vs Crawling)
            status = "IDLE"
            color = (0, 255, 0)

            if aspect > CRAWL_RATIO:
                if feet_y > floor_thresh:
                    status = "CRAWLING"
                    color = (0, 0, 255) # Red
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ALERT] {uid} is CRAWLING!")
                else:
                    status = "SITTING" 
                    color = (0, 255, 255) # Yellow

            # 2. Motion Detection (Running)
            if uid in entity_memory:
                prev = entity_memory[uid]
                prev_area = prev.get('area', 0)
                prev_cx, prev_cy = prev.get('center', (0,0))
                
                # Speed
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                if (prev_area > 0 and (w*h) > prev_area * RUN_GROWTH) or dist > RUN_SPEED_PIXELS:
                    status = "RUNNING"
                    color = (0, 0, 255)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ALERT] {uid} is RUNNING! Speed:{dist:.1f}")
                elif dist > WALK_SPEED_PIXELS and status != "CRAWLING":
                    status = "WALKING"

            # 3. Update Memory
            if uid not in entity_memory:
                entity_memory[uid] = {'face': '...', 'pose': '...'}
            
            entity_memory[uid]['area'] = w*h
            entity_memory[uid]['center'] = (cx, cy)
            
            # 4. DISPATCH to Workers (Only if face/pose is unknown or pending)
            # We send every 10th frame to avoid spamming disk I/O
            if frame_cnt % 5 == 0:
                crop = ai_frames[i][max(0,y1):min(AI_H,y2), max(0,x1):min(AI_W,x2)]
                if crop.size > 0:
                    if cam_name == "DOOR": dispatch("face", crop, uid)
                    if cam_name == "GATE": dispatch("pose", crop, uid)

            # --- DISPLAY ---
            face_info = entity_memory[uid].get('face', '...')
            pose_info = entity_memory[uid].get('pose', '...')
            
            # Override color for Identity
            if face_info == "puneeth": color = (0, 255, 0) # Verified
            elif face_info == "Unknown": color = (0, 0, 255) # Stranger
            
            # Override for Violence
            if "VIOLENCE" in pose_info: color = (0, 0, 255)

            label = f"{uid} | {status} | {face_info}"
            if "VIOLENCE" in pose_info: label += " | VIOLENCE"
            
            cv2.rectangle(disp, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), color, 2)
            cv2.putText(disp, label, (int(x1*sx), int(y1*sy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        display_frames.append(disp)

    combined = cv2.hconcat(display_frames)
    if active_weapons:
         cv2.putText(combined, f"WEAPON: {active_weapons[0]}", (50, 100), 1, 4, (0,0,255), 4)

    cv2.imshow("Phase 2: Integrated", combined)
    if cv2.waitKey(1) == ord('q'): break

cv2.destroyAllWindows()