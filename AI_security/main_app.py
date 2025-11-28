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

# Configuration
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
YOLO_MODEL_PATH = "yolov8n.pt"

AI_WIDTH, AI_HEIGHT = 640, 360
DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540
PROCESS_EVERY_N_FRAMES = 3

# Enhanced weapon detection with grouped classes
WEAPON_CLASSES = [43, 76, 34]  # scissors, scissors, knife
WEAPON_CONFIDENCE = 0.3  # Slightly higher for fewer false positives
WEAPON_PERSISTENCE_FRAMES = 5  # Frames to keep weapon alert

DIRS = ["jobs_face", "results_face", "jobs_pose", "results_pose"]

# GPU setup
gpu = torch.cuda.is_available()
device = 0 if gpu else 'cpu'
model = YOLO(YOLO_MODEL_PATH)
model.to(device)

# Trackers
gate_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
door_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Queues
q_gate = queue.Queue(maxsize=2)
q_door = queue.Queue(maxsize=2)

# World state with enhanced tracking
world_state = {}
weapon_detections = {}  # Track weapon persistence

def cleanup():
    """Periodic cleanup of old job/result files"""
    while True:
        time.sleep(30)
        now = time.time()
        for d in DIRS:
            if os.path.exists(d):
                for f in os.listdir(d):
                    try:
                        filepath = os.path.join(d, f)
                        if os.path.getmtime(filepath) < now - 30:
                            os.remove(filepath)
                    except:
                        pass

def capture(url, q):
    """Capture frames from RTSP stream"""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            continue
        
        if q.full():
            try:
                q.get_nowait()
            except:
                pass
        q.put(frame)

def dispatch(env, img, jid, metadata=None):
    """Dispatch job to worker with optional metadata"""
    try:
        d = "jobs_face" if env == 'face_env' else "jobs_pose"
        
        # Save image
        cv2.imwrite(f"{d}/{jid}.jpg", img)
        
        # Save metadata if provided (for face quality info)
        if metadata:
            with open(f"{d}/{jid}_meta.txt", "w") as f:
                for key, val in metadata.items():
                    f.write(f"{key}:{val}\n")
    except Exception as e:
        print(f"Dispatch error: {e}")

def calculate_face_quality(crop):
    """Calculate basic quality metrics for face crop"""
    # Size check
    h, w = crop.shape[:2]
    if h < 80 or w < 80:
        return {'size_ok': False, 'blur_score': 0, 'brightness': 0}
    
    # Blur detection using Laplacian variance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness check
    brightness = np.mean(gray)
    
    return {
        'size_ok': True,
        'blur_score': blur_score,
        'brightness': brightness,
        'width': w,
        'height': h
    }

def check_results():
    """Check and process worker results"""
    # Face recognition results
    for f in glob.glob("results_face/*.txt"):
        try:
            jid = os.path.basename(f).replace('result_', '').replace('.txt', '')
            with open(f, 'r') as file:
                res = file.read().strip()
            
            if res and res != "Error" and jid in world_state:
                world_state[jid]['identity'] = res
                
                # Parse confidence if available
                if '|' in res:
                    name, confidence = res.split('|')
                    world_state[jid]['identity'] = name
                    world_state[jid]['confidence'] = float(confidence)
            
            os.remove(f)
        except Exception as e:
            print(f"Error reading face result: {e}")
    
    # Pose detection results
    for f in glob.glob("results_pose/*.txt"):
        try:
            jid = os.path.basename(f).replace('result_', '').replace('.txt', '')
            with open(f, 'r') as file:
                res = file.read().strip()
            
            if res and res != "Error" and jid in world_state:
                world_state[jid]['pose'] = res
                
                # Track violence alerts
                if "VIOLENCE" in res or "WEAPON" in res:
                    world_state[jid]['alert'] = True
            
            os.remove(f)
        except Exception as e:
            print(f"Error reading pose result: {e}")

def update_weapon_tracking(weapon_name, frame_num):
    """Track weapon detections over time for persistence"""
    if weapon_name not in weapon_detections:
        weapon_detections[weapon_name] = []
    
    weapon_detections[weapon_name].append(frame_num)
    
    # Keep only recent detections
    weapon_detections[weapon_name] = [
        f for f in weapon_detections[weapon_name] 
        if frame_num - f <= WEAPON_PERSISTENCE_FRAMES
    ]

def is_weapon_detected(frame_num):
    """Check if any weapon has been consistently detected"""
    for weapon, frames in weapon_detections.items():
        frames = [f for f in frames if frame_num - f <= WEAPON_PERSISTENCE_FRAMES]
        if len(frames) >= 2:  # At least 2 detections in recent frames
            return True, weapon
    return False, None

# Initialize directories
for d in DIRS:
    if not os.path.exists(d):
        os.makedirs(d)
    # Clean existing files
    for f in glob.glob(f"{d}/*"):
        try:
            os.remove(f)
        except:
            pass

# Start threads
threading.Thread(target=capture, args=(RTSP_URL_GATE, q_gate), daemon=True).start()
threading.Thread(target=capture, args=(RTSP_URL_DOOR, q_door), daemon=True).start()
threading.Thread(target=cleanup, daemon=True).start()

print("AI Security System Started")
print(f"GPU: {gpu}")

frame_cnt = 0
cached_gate = []
cached_door = []

while True:
    try:
        img_gate = q_gate.get(timeout=1)
        img_door = q_door.get(timeout=1)
    except:
        continue
    
    frame_cnt += 1
    ai_gate = cv2.resize(img_gate, (AI_WIDTH, AI_HEIGHT))
    ai_door = cv2.resize(img_door, (AI_WIDTH, AI_HEIGHT))
    
    # Process every N frames
    if frame_cnt % PROCESS_EVERY_N_FRAMES == 0:
        classes = [0] + WEAPON_CLASSES
        res_gate = model(ai_gate, classes=classes, verbose=False, conf=WEAPON_CONFIDENCE)[0]
        res_door = model(ai_door, classes=classes, verbose=False, conf=WEAPON_CONFIDENCE)[0]
        
        gate_boxes, door_boxes = [], []
        
        def parse(res, box_list):
            if res.boxes:
                for box in res.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    
                    if cls == 0:  # Person
                        box_list.append([x1, y1, x2, y2, conf])
                    elif cls in WEAPON_CLASSES:
                        weapon_name = model.names[cls]
                        update_weapon_tracking(weapon_name, frame_cnt)
        
        parse(res_gate, gate_boxes)
        parse(res_door, door_boxes)
        
        # Update trackers
        cached_gate = gate_tracker.update(
            np.array([[*b[:4], 1] for b in gate_boxes]) if gate_boxes else np.empty((0, 5))
        )
        cached_door = door_tracker.update(
            np.array([[*b[:4], 1] for b in door_boxes]) if door_boxes else np.empty((0, 5))
        )
        
        # Dispatch jobs for gate (pose detection)
        for d in cached_gate:
            jid = f"gate_{int(d[4])}"
            if jid not in world_state:
                world_state[jid] = {'pose': 'Pending...', 'camera': 'gate'}
                x1, y1, x2, y2 = map(int, d[:4])
                crop = ai_gate[max(0, y1):min(AI_HEIGHT, y2), max(0, x1):min(AI_WIDTH, x2)]
                if crop.size > 0:
                    dispatch('pose_env', crop, jid)
        
        # Dispatch jobs for door (face recognition with quality check)
        for d in cached_door:
            jid = f"door_{int(d[4])}"
            if jid not in world_state:
                x1, y1, x2, y2 = map(int, d[:4])
                crop = ai_door[max(0, y1):min(AI_HEIGHT, y2), max(0, x1):min(AI_WIDTH, x2)]
                
                if crop.size > 0:
                    # Calculate face quality
                    quality = calculate_face_quality(crop)
                    
                    world_state[jid] = {
                        'identity': 'Pending...',
                        'camera': 'door',
                        'quality': quality
                    }
                    
                    # Dispatch with metadata
                    dispatch('face_env', crop, jid, metadata=quality)
    
    # Check for results
    check_results()
    
    # Display logic
    disp_gate = cv2.resize(img_gate, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    disp_door = cv2.resize(img_door, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    # Scale factors for drawing
    scale_x_gate = DISPLAY_WIDTH / AI_WIDTH
    scale_y_gate = DISPLAY_HEIGHT / AI_HEIGHT
    scale_x_door = DISPLAY_WIDTH / AI_WIDTH
    scale_y_door = DISPLAY_HEIGHT / AI_HEIGHT
    
    # Draw gate boxes
    for d in cached_gate:
        x1, y1, x2, y2 = map(int, d[:4])
        jid = f"gate_{int(d[4])}"
        
        # Scale coordinates
        x1, x2 = int(x1 * scale_x_gate), int(x2 * scale_x_gate)
        y1, y2 = int(y1 * scale_y_gate), int(y2 * scale_y_gate)
        
        # Color based on pose
        color = (0, 255, 0)
        if jid in world_state:
            pose = world_state[jid].get('pose', 'Pending...')
            if 'VIOLENCE' in pose:
                color = (0, 0, 255)
            elif 'SURRENDER' in pose:
                color = (0, 255, 255)
        
        cv2.rectangle(disp_gate, (x1, y1), (x2, y2), color, 2)
        
        if jid in world_state:
            pose = world_state[jid].get('pose', 'Pending...')
            cv2.putText(disp_gate, pose, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw door boxes
    for d in cached_door:
        x1, y1, x2, y2 = map(int, d[:4])
        jid = f"door_{int(d[4])}"
        
        # Scale coordinates
        x1, x2 = int(x1 * scale_x_door), int(x2 * scale_x_door)
        y1, y2 = int(y1 * scale_y_door), int(y2 * scale_y_door)
        
        # Color based on identity
        color = (0, 255, 0)
        if jid in world_state:
            identity = world_state[jid].get('identity', 'Pending...')
            if identity == 'Unknown':
                color = (0, 0, 255)
        
        cv2.rectangle(disp_door, (x1, y1), (x2, y2), color, 2)
        
        if jid in world_state:
            identity = world_state[jid].get('identity', 'Pending...')
            cv2.putText(disp_door, identity, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Weapon alert
    weapon_active, weapon_type = is_weapon_detected(frame_cnt)
    if weapon_active:
        alert_text = f"WEAPON DETECTED: {weapon_type.upper()}"
        cv2.rectangle(disp_gate, (10, 10), (DISPLAY_WIDTH - 10, 60), (0, 0, 255), -1)
        cv2.putText(disp_gate, alert_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Combine displays
    combined = cv2.hconcat([disp_gate, disp_door])
    
    # Info overlay
    cv2.putText(combined, f"Frame: {frame_cnt}", (10, DISPLAY_HEIGHT - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("AI Security System", combined)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()