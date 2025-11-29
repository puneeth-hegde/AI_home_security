# main_app_v2.py - SENTINEL BRAIN (FIXED & ENHANCED)
import cv2
import os
import time
import threading
import queue
import numpy as np
import glob
import torch
import json
from ultralytics import YOLO
from sort import Sort
from datetime import datetime
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
YOLO_MODEL_PATH = "yolov8n.pt"

AI_WIDTH, AI_HEIGHT = 640, 360
DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540

# Audio queue directory
AUDIO_QUEUE_DIR = "audio_queue"
RESP_FILE = "audio_resp.txt"

# Authorized users
AUTHORIZED_USERS = ["puneeth"]

# Movement thresholds
LATERAL_RUN_SPEED = 65.0
Z_APPROACH_RATE = 0.40
CHARGING_THRESHOLD = 50.0
WALK_SPEED_MIN = 15.0

# Posture thresholds (4-SIGNAL LOGIC)
CRAWL_ASPECT_MIN = 1.6
CRAWL_CENTROID_MIN = 0.70
CRAWL_HEIGHT_MAX = 0.35
CRAWL_HIPS_MIN = 0.75

# Weapon detection
WEAPON_CLASSES = [43, 76]  # knife, scissors
WEAPON_CONF = 0.25
WEAPON_PERSIST_FRAMES = 5
WEAPON_HAND_DIST_NORMALIZED = 0.15  # 15% of image width

# Package classes
PACKAGE_CLASSES = [24, 25, 26, 27, 28]

# Hostage detection
HOSTAGE_DISTANCE_THRESHOLD = 100  # pixels at 640x360
HOSTAGE_CHECK_INTERVAL = 10  # frames

# ============================================================================
# LOGGING SYSTEM
# ============================================================================
os.makedirs("logs", exist_ok=True)

class SentinelLogger:
    def __init__(self):
        self.det = logging.getLogger('detection')
        det_handler = logging.FileHandler('logs/detection.log')
        det_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        self.det.addHandler(det_handler)
        self.det.setLevel(logging.INFO)
        
        self.threat = logging.getLogger('threat')
        threat_handler = logging.FileHandler('logs/threat.log')
        threat_handler.setFormatter(logging.Formatter('[%(asctime)s] [ALERT] %(message)s'))
        self.threat.addHandler(threat_handler)
        self.threat.setLevel(logging.WARNING)
        
        self.perf = logging.getLogger('performance')
        perf_handler = logging.FileHandler('logs/performance.log')
        perf_handler.setFormatter(logging.Formatter('[%(asctime)s] [PERF] %(message)s'))
        self.perf.addHandler(perf_handler)
        self.perf.setLevel(logging.DEBUG)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] %(message)s'))
        self.det.addHandler(console)
        self.threat.addHandler(console)
    
    def log_detection(self, cam, pid, event, conf, meta):
        msg = f"[{cam}] [{pid}] {event} conf={conf:.2f} meta={json.dumps(meta)}"
        self.det.info(msg)
    
    def log_threat(self, threat_type, pid, evidence):
        msg = f"[{threat_type}] pid={pid} evidence={json.dumps(evidence)}"
        self.threat.warning(msg)
    
    def log_perf(self, component, metric, value):
        msg = f"[{component}] {metric}={value:.3f}"
        self.perf.debug(msg)

logger = SentinelLogger()

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================
print("=" * 80)
print("SENTINEL v3.0 - CONTEXT-AWARE SECURITY SYSTEM (FIXED)")
print("=" * 80)

gpu = torch.cuda.is_available()
device = 0 if gpu else 'cpu'
print(f"[SYSTEM] Hardware: {'GPU (CUDA)' if gpu else 'CPU'}")
logger.log_detection("SYSTEM", "INIT", "HARDWARE", 1.0, {"device": str(device)})

model = YOLO(YOLO_MODEL_PATH)
model.to(device)

gate_tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
door_tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)

q_gate = queue.Queue(maxsize=2)
q_door = queue.Queue(maxsize=2)

# State management
person_memory = {}
weapon_evidence = {}
last_verified_time = 0
frame_cnt = 0
command_counter = 0  # For unique command IDs

# Setup audio queue
os.makedirs(AUDIO_QUEUE_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def send_cmd(text, priority=2):
    """
    Send command to audio queue with priority
    Priority: 0=URGENT, 1=HIGH, 2=NORMAL, 3=LOW
    """
    global command_counter
    try:
        timestamp = int(time.time() * 1000)
        command_counter += 1
        # Format: priority_timestamp_counter.txt
        filename = f"{priority}_{timestamp}_{command_counter}.txt"
        filepath = os.path.join(AUDIO_QUEUE_DIR, filename)
        
        with open(filepath, "w") as f:
            f.write(text)
        
        logger.log_detection("AUDIO", "CMD", text, 1.0, {"priority": priority})
    except Exception as e:
        logger.log_threat("AUDIO_FAILURE", "SYSTEM", {"error": str(e)})

def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def bbox_width(bbox):
    return bbox[2] - bbox[0]

def bbox_height(bbox):
    return bbox[3] - bbox[1]

# ============================================================================
# MOVEMENT ANALYSIS
# ============================================================================
def analyze_movement(person_id, current_bbox, prev_bbox, dt, frame_dims):
    """
    Separates lateral (X/Y) from Z-axis (depth) movement
    Returns: (status, confidence, metadata)
    """
    if dt < 0.001:
        return "IDLE", 0.5, {}
    
    cx, cy = bbox_center(current_bbox)
    px, py = bbox_center(prev_bbox)
    
    lateral_dist = np.hypot(cx - px, cy - py)
    lateral_speed = lateral_dist / dt
    
    curr_area = bbox_area(current_bbox)
    prev_area = bbox_area(prev_bbox)
    
    if prev_area > 0:
        z_growth_rate = (curr_area / prev_area - 1.0) / dt
    else:
        z_growth_rate = 0.0
    
    is_running = lateral_speed > LATERAL_RUN_SPEED
    is_charging = (lateral_speed > CHARGING_THRESHOLD) and (z_growth_rate > Z_APPROACH_RATE * 0.7)
    is_walking = lateral_speed > WALK_SPEED_MIN and lateral_speed < LATERAL_RUN_SPEED
    
    metadata = {
        "lateral_speed": round(lateral_speed, 2),
        "z_growth": round(z_growth_rate, 3),
        "dt": round(dt, 3)
    }
    
    if is_charging:
        logger.log_threat("CHARGING_DETECTED", person_id, metadata)
        return "CHARGING", 0.95, metadata
    elif is_running:
        logger.log_threat("RUNNING_DETECTED", person_id, metadata)
        return "RUNNING", 0.90, metadata
    elif is_walking:
        return "WALKING", 0.80, metadata
    else:
        return "IDLE", 0.70, metadata

# ============================================================================
# POSTURE ANALYSIS (4-SIGNAL FUSION)
# ============================================================================
def analyze_posture(person_id, bbox, frame_height, pose_hints):
    """
    4-Signal Logic: aspect, centroid, height, hips
    Returns: (status, confidence, metadata)
    """
    x1, y1, x2, y2 = bbox
    width = bbox_width(bbox)
    height = bbox_height(bbox)
    
    if height == 0:
        return "IDLE", 0.5, {}
    
    aspect_ratio = width / height
    centroid_y = (y1 + y2) / 2 / frame_height
    bbox_height_ratio = height / frame_height
    
    # Get hips position from pose worker
    hips_y = pose_hints.get('hips_y', 0.5)
    
    metadata = {
        "aspect": round(aspect_ratio, 2),
        "centroid_y": round(centroid_y, 2),
        "bbox_height": round(bbox_height_ratio, 2),
        "hips_y": round(hips_y, 2)
    }
    
    # Count signals
    signals = 0
    
    if aspect_ratio > CRAWL_ASPECT_MIN:
        signals += 1
        metadata['signal_aspect'] = 1
    
    if centroid_y > CRAWL_CENTROID_MIN:
        signals += 1
        metadata['signal_centroid'] = 1
    
    if bbox_height_ratio < CRAWL_HEIGHT_MAX:
        signals += 1
        metadata['signal_height'] = 1
    
    if hips_y > CRAWL_HIPS_MIN:
        signals += 1
        metadata['signal_hips'] = 1
    
    metadata['signals'] = signals
    
    # Need 3+ signals for crawling
    if signals >= 3:
        logger.log_threat("CRAWLING_POSTURE", person_id, metadata)
        return "CRAWLING", 0.90, metadata
    
    return "IDLE", 0.85, metadata

# ============================================================================
# WEAPON VALIDATION
# ============================================================================
def validate_weapon(person_id, weapon_class, frame_no, weapon_bbox, pose_hints, img_width):
    """
    Validates weapon detection using:
    1. Temporal persistence (must see in multiple frames)
    2. Hand proximity (weapon near wrist)
    Returns: (is_valid, reason)
    """
    if person_id not in weapon_evidence:
        weapon_evidence[person_id] = {}
    
    weapon_name = model.names[weapon_class]
    
    if weapon_name not in weapon_evidence[person_id]:
        weapon_evidence[person_id][weapon_name] = []
    
    weapon_evidence[person_id][weapon_name].append({
        'frame': frame_no,
        'bbox': weapon_bbox
    })
    
    recent = [e for e in weapon_evidence[person_id][weapon_name] if frame_no - e['frame'] < 20]
    weapon_evidence[person_id][weapon_name] = recent
    
    # Check persistence
    if len(recent) < WEAPON_PERSIST_FRAMES:
        return False, f"Insufficient persistence: {len(recent)}/{WEAPON_PERSIST_FRAMES}"
    
    # Check hand proximity (if pose data available)
    if 'left_wrist_x' in pose_hints and 'right_wrist_x' in pose_hints:
        weapon_center_x = (weapon_bbox[0] + weapon_bbox[2]) / 2
        weapon_center_y = (weapon_bbox[1] + weapon_bbox[3]) / 2
        
        left_wrist = (pose_hints['left_wrist_x'] * img_width, pose_hints['left_wrist_y'] * img_width)
        right_wrist = (pose_hints['right_wrist_x'] * img_width, pose_hints['right_wrist_y'] * img_width)
        
        dist_left = np.hypot(weapon_center_x - left_wrist[0], weapon_center_y - left_wrist[1])
        dist_right = np.hypot(weapon_center_x - right_wrist[0], weapon_center_y - right_wrist[1])
        
        min_dist = min(dist_left, dist_right)
        threshold_dist = img_width * WEAPON_HAND_DIST_NORMALIZED
        
        if min_dist > threshold_dist:
            return False, f"Too far from hands: {min_dist:.1f}px > {threshold_dist:.1f}px"
        
        return True, f"Near hand ({min_dist:.1f}px), {len(recent)} frames"
    else:
        # No pose data, rely on persistence only
        return True, f"Persistent detection: {len(recent)} frames (no pose data)"

# ============================================================================
# HOSTAGE DETECTION
# ============================================================================
def detect_hostage_situation(verified_persons, unknown_persons, camera_type):
    """
    Detects if an unknown person is dangerously close to a verified user
    Camera-aware: door camera (top-down) vs gate camera (frontal)
    Returns: (is_hostage, evidence)
    """
    for v_id, v_data in verified_persons.items():
        v_center = v_data['center']
        v_cam = 'door' if 'door' in v_id else 'gate'
        
        for u_id, u_data in unknown_persons.items():
            u_center = u_data['center']
            u_cam = 'door' if 'door' in u_id else 'gate'
            
            # Must be same camera
            if v_cam != u_cam:
                continue
            
            # Calculate distance
            distance = np.hypot(v_center[0] - u_center[0], v_center[1] - u_center[1])
            
            if distance < HOSTAGE_DISTANCE_THRESHOLD:
                # Check if unknown is "behind" based on camera type
                is_behind = False
                
                if v_cam == 'door':  # Top-down camera
                    # "Behind" means higher Y (further from door)
                    if u_center[1] < v_center[1] - 30:
                        is_behind = True
                else:  # Gate camera (frontal)
                    # "Behind" means similar Y but further X
                    if abs(u_center[1] - v_center[1]) < 50 and u_center[0] > v_center[0] + 40:
                        is_behind = True
                
                if is_behind:
                    evidence = {
                        "verified_person": v_id,
                        "unknown_person": u_id,
                        "distance": round(distance, 1),
                        "camera": v_cam,
                        "verified_pos": [round(v_center[0], 1), round(v_center[1], 1)],
                        "unknown_pos": [round(u_center[0], 1), round(u_center[1], 1)]
                    }
                    return True, evidence
    
    return False, {}

# ============================================================================
# WORKER RESULT CHECKING
# ============================================================================
def check_results():
    """Check and process results from face and pose workers"""
    
    # FACE RESULTS
    face_results = glob.glob("results_face/result_*.txt")
    for result_file in face_results:
        try:
            job_id = os.path.basename(result_file).replace("result_", "").replace(".txt", "")
            
            with open(result_file, 'r') as f:
                res = f.read().strip()
            
            if job_id in person_memory:
                mem = person_memory[job_id]
                mem['face'] = res
                
                # Anti-spoof handling
                if "ERROR:SPOOF" in res:
                    logger.log_threat("SPOOF_DETECTED", job_id, {"reason": res})
                    send_cmd("STEP_CLOSER", priority=2)
                    mem['status'] = 'SUSPECTED_SPOOF'
                
                # Valid identity
                elif res not in ["Unknown", "Verifying...", "ERROR:BLURRY", "ERROR:DARK", "ERROR:BRIGHT", "ERROR:FLAT", "Error"]:
                    global last_verified_time
                    last_verified_time = time.time()
                    if mem['status'] != 'VERIFIED':
                        mem['status'] = 'VERIFIED'
                        mem['identity'] = res
                        logger.log_detection(mem.get('camera', 'UNKNOWN'), job_id, "VERIFIED", 0.95, {"identity": res})
                        send_cmd(f"WELCOME:{res}", priority=2)
                
                # Unknown but stable
                elif res == "Unknown":
                    if time.time() - last_verified_time > 15.0:
                        if time.time() - mem['start_time'] > 8.0 and not mem.get('warned', False):
                            logger.log_detection(mem.get('camera', 'UNKNOWN'), job_id, "STRANGER_LINGERING", 0.80, {})
                            mem['warned'] = True
                            send_cmd("ASK_NAME", priority=2)
                
                # Quality issues
                elif res == "ERROR:BLURRY":
                    send_cmd("FIX_BLUR", priority=3)
                elif res == "ERROR:DARK":
                    send_cmd("FIX_DARK", priority=3)
                elif res == "ERROR:BRIGHT":
                    send_cmd("FIX_BRIGHT", priority=3)
            
            os.remove(result_file)
        except Exception as e:
            logger.log_threat("RESULT_PARSE_ERROR", "face", {"error": str(e)})
    
    # POSE RESULTS
    pose_results = glob.glob("results_pose/result_*.txt")
    for result_file in pose_results:
        try:
            job_id = os.path.basename(result_file).replace("result_", "").replace(".txt", "")
            
            with open(result_file, 'r') as f:
                result_text = f.read().strip()
            
            if job_id in person_memory:
                # Parse: "VIOLENCE|1.85|hips_y=0.45,left_wrist_x=0.3,..."
                parts = result_text.split('|')
                status = parts[0]
                velocity = float(parts[1]) if len(parts) > 1 else 0.0
                
                # Parse metadata
                metadata = {}
                if len(parts) >= 3 and parts[2]:
                    for kv in parts[2].split(','):
                        if '=' in kv:
                            k, v = kv.split('=')
                            try:
                                metadata[k] = float(v)
                            except:
                                metadata[k] = v
                
                person_memory[job_id]['pose'] = status
                person_memory[job_id]['pose_raw'] = result_text
                person_memory[job_id]['pose_hints'] = metadata
                person_memory[job_id]['pose_velocity'] = velocity
                
                # Check for surrender (hands up with package check)
                if status == "SURRENDER":
                    # TODO: Check if holding package before triggering alarm
                    logger.log_detection(person_memory[job_id].get('camera', 'UNKNOWN'), 
                                        job_id, "SURRENDER_DETECTED", 0.90, metadata)
            
            os.remove(result_file)
        except Exception as e:
            logger.log_threat("RESULT_PARSE_ERROR", "pose", {"error": str(e), "file": result_file})

# ============================================================================
# WORKER DISPATCH
# ============================================================================
def dispatch(env, img, jid):
    try:
        folder = 'jobs_face' if env == 'face' else 'jobs_pose'
        os.makedirs(folder, exist_ok=True)
        path = f"{folder}/{jid}.jpg"
        cv2.imwrite(path, img)
    except Exception as e:
        logger.log_threat("DISPATCH_ERROR", jid, {"error": str(e)})

# ============================================================================
# CAMERA CAPTURE THREADS
# ============================================================================
def capture(url, q, name):
    logger.log_detection(name, "THREAD", "STARTING", 1.0, {})
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue
        
        if q.full():
            try:
                q.get_nowait()
            except:
                pass
        q.put(frame)

threading.Thread(target=capture, args=(RTSP_URL_GATE, q_gate, "GATE"), daemon=True).start()
threading.Thread(target=capture, args=(RTSP_URL_DOOR, q_door, "DOOR"), daemon=True).start()

time.sleep(3)

# ============================================================================
# MAIN LOOP
# ============================================================================
logger.log_detection("SYSTEM", "MAIN", "STARTING", 1.0, {})
loop_start = time.time()

while True:
    frame_cnt += 1
    
    try:
        img_g = q_gate.get(timeout=1)
        img_d = q_door.get(timeout=1)
    except:
        continue
    
    # Resize for AI processing
    ai_g = cv2.resize(img_g, (AI_WIDTH, AI_HEIGHT))
    ai_d = cv2.resize(img_d, (AI_WIDTH, AI_HEIGHT))
    
    # YOLO inference every 3rd frame
    if frame_cnt % 3 == 0:
        yolo_start = time.time()
        
        res_g = model(ai_g, classes=[0] + WEAPON_CLASSES + PACKAGE_CLASSES, 
                      verbose=False, conf=WEAPON_CONF)[0]
        res_d = model(ai_d, classes=[0] + WEAPON_CLASSES + PACKAGE_CLASSES, 
                      verbose=False, conf=WEAPON_CONF)[0]
        
        logger.log_perf("YOLO", "inference_ms", (time.time() - yolo_start) * 1000)
        
        def parse_detections(res, tracker, cam_type, img_src):
            boxes = []
            weapons_detected = []
            packages_detected = []
            
            for box in res.boxes:
                cls = int(box.cls[0])
                coords = box.xyxy[0].cpu().numpy()
                
                if cls == 0:
                    boxes.append(coords)
                elif cls in WEAPON_CLASSES:
                    weapons_detected.append({
                        'class': cls,
                        'bbox': coords,
                        'name': model.names[cls]
                    })
                elif cls in PACKAGE_CLASSES:
                    packages_detected.append({
                        'class': cls,
                        'bbox': coords
                    })
            
            tracks = tracker.update(np.array(boxes) if boxes else np.empty((0, 5)))
            
            for d in tracks:
                x1, y1, x2, y2, tid = map(int, d)
                jid = f"{cam_type}_{tid}"
                
                # Initialize memory
                if jid not in person_memory:
                    person_memory[jid] = {
                        'camera': cam_type,
                        'start_time': time.time(),
                        'timestamp': time.time(),
                        'last_box': [x1, y1, x2, y2],
                        'last_time': time.time(),
                        'center': bbox_center([x1, y1, x2, y2]),
                        'pose': 'Normal',
                        'pose_raw': '',
                        'pose_hints': {},
                        'face': '...',
                        'identity': 'Unknown',
                        'status': 'IDLE',
                        'warned': False,
                        'movement': 'IDLE',
                        'posture': 'IDLE',
                        'pose_velocity': 0.0
                    }
                
                mem = person_memory[jid]
                
                # MOVEMENT ANALYSIS
                prev_box = mem['last_box']
                prev_time = mem['last_time']
                now_time = time.time()
                dt = now_time - prev_time
                
                movement_status, move_conf, move_meta = analyze_movement(
                    jid, [x1, y1, x2, y2], prev_box, dt, (AI_WIDTH, AI_HEIGHT)
                )
                mem['movement'] = movement_status
                
                # POSTURE ANALYSIS (using fresh pose_hints from worker)
                posture_status, post_conf, post_meta = analyze_posture(
                    jid, [x1, y1, x2, y2], AI_HEIGHT, mem['pose_hints']
                )
                mem['posture'] = posture_status
                
                # THREAT EVALUATION
                # Crawling always triggers alarm
                if posture_status == "CRAWLING":
                    logger.log_threat("CRAWLER_ALARM", jid, post_meta)
                    send_cmd("WARN_CRAWLING", priority=1)
                
                # Running/Charging
                if movement_status in ["RUNNING", "CHARGING"]:
                    if mem['identity'] not in AUTHORIZED_USERS or movement_status == "CHARGING":
                        logger.log_threat("RUNNER_ALARM", jid, move_meta)
                        send_cmd("WARN_RUNNING", priority=1)
                
                # Violence detection
                if mem['pose'] == "VIOLENCE" and mem['pose_velocity'] > 1.5:
                    logger.log_threat("VIOLENCE_ALARM", jid, {"velocity": mem['pose_velocity']})
                    send_cmd("WARN_INTRUDER", priority=0)
                
                # Weapon validation
                for weapon in weapons_detected:
                    is_valid, reason = validate_weapon(
                        jid, weapon['class'], frame_cnt, weapon['bbox'], 
                        mem.get('pose_hints', {}), AI_WIDTH
                    )
                    if is_valid:
                        logger.log_threat("WEAPON_ALARM", jid, {
                            "weapon": weapon['name'],
                            "reason": reason
                        })
                        send_cmd("WARN_WEAPON", priority=0)
                
                # Log state
                if frame_cnt % 30 == 0:  # Every second at 30fps
                    logger.log_detection(cam_type, jid, "STATE_UPDATE", 0.90, {
                        "movement": movement_status,
                        "posture": posture_status,
                        "identity": mem['identity'],
                        "face": mem['face']
                    })
                
                # Update memory
                mem['last_box'] = [x1, y1, x2, y2]
                mem['last_time'] = now_time
                mem['center'] = bbox_center([x1, y1, x2, y2])
                mem['timestamp'] = now_time
                
                # Dispatch crops to workers (BOTH CAMERAS GET BOTH WORKERS)
                crop = img_src[max(0, y1):min(AI_HEIGHT, y2), max(0, x1):min(AI_WIDTH, x2)]
                
                if crop.size > 0:
                    # Pose every 2nd frame (15 FPS) - BOTH CAMERAS
                    if frame_cnt % 2 == 0:
                        dispatch('pose', crop, jid)
                    
                    # Face every 5th frame (6 FPS) - BOTH CAMERAS
                    if frame_cnt % 5 == 0:
                        dispatch('face', crop, jid)
        
        parse_detections(res_g, gate_tracker, 'gate', ai_g)
        parse_detections(res_d, door_tracker, 'door', ai_d)
    
    # Check worker results
    check_results()
    
    # Hostage detection (every 10 frames)
    if frame_cnt % HOSTAGE_CHECK_INTERVAL == 0:
        verified = {k: v for k, v in person_memory.items() if v['identity'] in AUTHORIZED_USERS}
        unknown = {k: v for k, v in person_memory.items() if v['identity'] == 'Unknown'}
        
        is_hostage, evidence = detect_hostage_situation(verified, unknown, "door")
        if is_hostage:
            logger.log_threat("HOSTAGE_SITUATION", "SYSTEM", evidence)
            send_cmd("SILENT_ALARM", priority=0)
    
    # DISPLAY
    disp_g = cv2.resize(img_g, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    disp_d = cv2.resize(img_d, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    sx, sy = DISPLAY_WIDTH / AI_WIDTH, DISPLAY_HEIGHT / AI_HEIGHT
    
    for jid, data in person_memory.items():
        if "last_box" in data:
            box = data['last_box']
            target = disp_g if "gate" in jid else disp_d
            
            identity = data.get('identity', 'Unknown')
            movement = data.get('movement', 'IDLE')
            posture = data.get('posture', 'IDLE')
            
            # Color coding
            if identity in AUTHORIZED_USERS:
                color = (0, 255, 0)
            elif posture == "CRAWLING" or movement in ["RUNNING", "CHARGING"]:
                color = (0, 0, 255)
            elif identity == "Unknown":
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)
            
            x1, y1 = int(box[0] * sx), int(box[1] * sy)
            x2, y2 = int(box[2] * sx), int(box[3] * sy)
            
            info = f"{identity} | {movement} | {posture}"
            
            cv2.rectangle(target, (x1, y1), (x2, y2), color, 2)
            cv2.putText(target, info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Combine displays
    comb = cv2.hconcat([disp_g, disp_d])
    
    # FPS overlay
    loop_time = time.time() - loop_start
    fps = frame_cnt / loop_time if loop_time > 0 else 0
    cv2.putText(comb, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("SENTINEL Command Center", comb)
    
    # Performance logging
    if frame_cnt % 100 == 0:
        logger.log_perf("MAIN", "loop_fps", fps)
        logger.log_perf("MAIN", "tracked_persons", len(person_memory))
    
    if cv2.waitKey(1) == ord('q'):
        logger.log_detection("SYSTEM", "MAIN", "SHUTDOWN", 1.0, {})
        break

cv2.destroyAllWindows()