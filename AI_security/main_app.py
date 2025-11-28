import cv2
import os
import time
from datetime import datetime
import threading
import queue
import numpy as np
import glob
import torch
from ultralytics import YOLO
from sort import Sort

# --- CONFIGURATION ---
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
YOLO_MODEL_PATH = "yolov8n.pt"

# --- DISPLAY & PERFORMANCE ---
AI_WIDTH, AI_HEIGHT = 640, 360       # Small for Speed
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720 # Big for Visibility
PROCESS_EVERY_N_FRAMES = 3

# --- WEAPON DETECTION ---
WEAPON_CLASSES = [43, 76, 34] # Knife, Scissors, Bat
WEAPON_CONFIDENCE = 0.25      # Sensitive for small objects

# --- PATHS ---
DIRS = ["jobs_face", "results_face", "jobs_pose", "results_pose"]

print("--- MAIN APPLICATION STARTING ---")
gpu = torch.cuda.is_available()
device = 0 if gpu else 'cpu'
print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU: {gpu}")

model = YOLO(YOLO_MODEL_PATH)
model.to(device)

gate_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
door_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

q_gate = queue.Queue(maxsize=2)
q_door = queue.Queue(maxsize=2)
stop_event = threading.Event()
world_state = {}

# --- WORKER THREADS ---
def cleanup():
    while not stop_event.is_set():
        time.sleep(30)
        now = time.time()
        for d in DIRS:
            if os.path.exists(d):
                for f in os.listdir(d):
                    try:
                        p = os.path.join(d, f)
                        if os.path.getmtime(p) < now - 30: os.remove(p)
                    except: pass

def capture(url, q, name):
    print(f"[{name}] Connecting...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.5)
            continue
        if q.full():
            try: q.get_nowait()
            except: pass
        q.put(frame)
    cap.release()

def dispatch(env, img, jid):
    try:
        d = "jobs_face" if env == 'face_env' else "jobs_pose"
        cv2.imwrite(f"{d}/{jid}.jpg", img)
    except: pass

def check_results():
    for f in glob.glob("results_face/*.txt"):
        try:
            jid = os.path.basename(f).replace('result_', '').replace('.txt', '')
            with open(f, 'r') as file: res = file.read().strip()
            if res and res != "Error" and jid in world_state:
                world_state[jid]['identity'] = res
            os.remove(f)
        except: pass
    for f in glob.glob("results_pose/*.txt"):
        try:
            jid = os.path.basename(f).replace('result_', '').replace('.txt', '')
            with open(f, 'r') as file: res = file.read().strip()
            if res and res != "Error" and jid in world_state:
                world_state[jid]['pose'] = res
            os.remove(f)
        except: pass

# --- STARTUP ---
for d in DIRS: 
    if not os.path.exists(d): os.makedirs(d)
    for f in glob.glob(f"{d}/*"): os.remove(f)

threading.Thread(target=capture, args=(RTSP_URL_GATE, q_gate, "GATE"), daemon=True).start()
threading.Thread(target=capture, args=(RTSP_URL_DOOR, q_door, "DOOR"), daemon=True).start()
threading.Thread(target=cleanup, daemon=True).start()

time.sleep(2) 

# --- MAIN LOOP ---
frame_cnt = 0
cached_gate_boxes = []
cached_door_boxes = []
detected_weapons = []
weapon_timer = 0

while not stop_event.is_set():
    try:
        img_gate = q_gate.get(timeout=1)
        img_door = q_door.get(timeout=1)
    except: continue

    frame_cnt += 1
    
    # Resize for AI
    ai_gate = cv2.resize(img_gate, (AI_WIDTH, AI_HEIGHT))
    ai_door = cv2.resize(img_door, (AI_WIDTH, AI_HEIGHT))

    # --- AI LOGIC ---
    if frame_cnt % PROCESS_EVERY_N_FRAMES == 0:
        detected_weapons = []
        classes = [0] + WEAPON_CLASSES
        
        res_gate = model(ai_gate, classes=classes, verbose=False, conf=WEAPON_CONFIDENCE)[0]
        res_door = model(ai_door, classes=classes, verbose=False, conf=WEAPON_CONFIDENCE)[0]

        gate_boxes, door_boxes = [], []

        def parse(res, box_list):
            if res.boxes:
                for box in res.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    if cls == 0: box_list.append([x1, y1, x2, y2])
                    elif cls in WEAPON_CLASSES:
                        name = model.names[cls]
                        detected_weapons.append(name)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ALERT] WEAPON: {name}")

        parse(res_gate, gate_boxes)
        parse(res_door, door_boxes)

        track_gate = gate_tracker.update(np.array(gate_boxes) if gate_boxes else np.empty((0, 5)))
        track_door = door_tracker.update(np.array(door_boxes) if door_boxes else np.empty((0, 5)))
        
        cached_gate_boxes = track_gate
        cached_door_boxes = track_door

        for d in track_gate:
            x1, y1, x2, y2, tid = map(int, d)
            jid = f"gate_{tid}"
            if jid not in world_state:
                world_state[jid] = {'identity': 'N/A', 'pose': 'Pending...'}
                dispatch('pose_env', ai_gate[max(0,y1):min(AI_HEIGHT,y2), max(0,x1):min(AI_WIDTH,x2)], jid)

        for d in track_door:
            x1, y1, x2, y2, tid = map(int, d)
            jid = f"door_{tid}"
            if jid not in world_state:
                world_state[jid] = {'identity': 'Pending...', 'pose': 'Pending...'}
                dispatch('face_env', ai_door[max(0,y1):min(AI_HEIGHT,y2), max(0,x1):min(AI_WIDTH,x2)], jid)

    check_results()

    # --- DRAWING ---
    # Resize Original HD frames for Display
    disp_gate = cv2.resize(img_gate, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    disp_door = cv2.resize(img_door, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    # Scale Ratio
    sx = DISPLAY_WIDTH / AI_WIDTH
    sy = DISPLAY_HEIGHT / AI_HEIGHT

    for d in cached_gate_boxes:
        x1, y1, x2, y2, tid = map(int, d)
        jid = f"gate_{tid}"
        pose = world_state.get(jid, {}).get('pose', '...')
        color = (0,0,255) if "Threat" in pose else (0,255,0)
        cv2.rectangle(disp_gate, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), color, 2)
        cv2.putText(disp_gate, pose, (int(x1*sx), int(y1*sy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for d in cached_door_boxes:
        x1, y1, x2, y2, tid = map(int, d)
        jid = f"door_{tid}"
        ident = world_state.get(jid, {}).get('identity', '...')
        color = (0,255,0) if ident == 'puneeth' else (0,0,255)
        cv2.rectangle(disp_door, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), color, 2)
        cv2.putText(disp_door, ident, (int(x1*sx), int(y1*sy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    combined = cv2.hconcat([disp_gate, disp_door])
    
    if detected_weapons: weapon_timer = 30
    if weapon_timer > 0:
        cv2.rectangle(combined, (0,0), (combined.shape[1], 80), (0,0,255), -1)
        cv2.putText(combined, "WEAPON DETECTED!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
        weapon_timer -= 1

    cv2.imshow("Security System (FINAL)", combined)
    if cv2.waitKey(1) == ord('q'): break

stop_event.set()
cv2.destroyAllWindows()