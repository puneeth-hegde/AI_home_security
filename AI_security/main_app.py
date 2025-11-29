# main_app.py - SENTINEL BRAIN (OPTIMIZED)
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

# Commands
CMD_FILE = "audio_cmd.txt"
RESP_FILE = "audio_resp.txt"

# Authorized users
AUTHORIZED_USERS = ["puneeth"]

# Movement thresholds (TUNED)
LATERAL_RUN_SPEED = 65.0      # pixels/sec in X/Y plane
Z_APPROACH_RATE = 0.40        # 40% bbox growth per second
CHARGING_THRESHOLD = 50.0     # lateral + z-axis combo
WALK_SPEED_MIN = 15.0         # minimum for "walking" classification

# Posture thresholds (4-SIGNAL LOGIC)
CRAWL_ASPECT_MIN = 1.6        # width/height ratio
CRAWL_CENTROID_MIN = 0.70     # normalized Y position (0=top, 1=bottom)
CRAWL_HEIGHT_MAX = 0.35       # bbox height as % of frame
CRAWL_HIPS_MIN = 0.75         # hip Y position from pose worker

# Weapon detection
WEAPON_CLASSES = [43, 76]     # knife, scissors (removed 34)
WEAPON_CONF = 0.25            # raised from 0.20 to reduce noise
WEAPON_PERSIST_FRAMES = 5     # must see weapon in 5 frames
WEAPON_HAND_DIST = 60         # pixels from wrist

# Package classes (for delivery detection)
PACKAGE_CLASSES = [24, 25, 26, 27, 28]  # backpack, umbrella, handbag, tie, suitcase

# ============================================================================
# LOGGING SYSTEM
# ============================================================================
os.makedirs("logs", exist_ok=True)

class SentinelLogger:
    def __init__(self):
        # Detection logger (all events)
        self.det = logging.getLogger('detection')
        det_handler = logging.FileHandler('logs/detection.log')
        det_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        self.det.addHandler(det_handler)
        self.det.setLevel(logging.INFO)
        
        # Threat logger (security only)
        self.threat = logging.getLogger('threat')
        threat_handler = logging.FileHandler('logs/threat.log')
        threat_handler.setFormatter(logging.Formatter('[%(asctime)s] [ALERT] %(message)s'))
        self.threat.addHandler(threat_handler)
        self.threat.setLevel(logging.WARNING)
        
        # Performance logger
        self.perf = logging.getLogger('performance')
        perf_handler = logging.FileHandler('logs/performance.log')
        perf_handler.setFormatter(logging.Formatter('[%(asctime)s] [PERF] %(message)s'))
        self.perf.addHandler(perf_handler)
        self.perf.setLevel(logging.DEBUG)
        
        # Console output
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
print("SENTINEL v3.0 - CONTEXT-AWARE SECURITY SYSTEM")
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
weapon_evidence = {}  # {person_id: {weapon_class: [frame_numbers]}}
last_verified_time = 0
frame_cnt = 0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def send_cmd(text):
    try:
        with open(CMD_FILE, "w") as f:
            f.write(text)
        logger.log_detection("AUDIO", "CMD", text, 1.0, {})
    except Exception as e:
        logger.log_threat("AUDIO_FAILURE", "SYSTEM", {"error": str(e)})

def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# ============================================================================
# MOVEMENT ANALYSIS (DIRECTIONAL)
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
    
    # Lateral velocity (actual running in X/Y plane)
    lateral_dist = np.hypot(cx - px, cy - py)
    lateral_speed = lateral_dist / dt
    
    # Z-axis growth (approaching camera)
    curr_area = bbox_area(current_bbox)
    prev_area = bbox_area(prev_bbox)
    
    if prev_area > 0:
        z_growth_rate = (curr_area / prev_area - 1.0) / dt  # % per second
    else:
        z_growth_rate = 0.0
    
    # CLASSIFICATION LOGIC
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
    w, h = x2 - x1, y2 - y1
    
    if h < 1:
        return "UNKNOWN", 0.0, {}
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Signal 1: Aspect Ratio
    aspect = w / h
    
    # Signal 2: Centroid Height (normalized)
    centroid_ratio = cy / frame_height
    
    # Signal 3: Bbox Height Ratio
    height_ratio = h / frame_height
    
    # Signal 4: Hip Position from pose_worker
    hips_y = pose_hints.get('hips_y', 0.5)
    
    metadata = {
        "aspect": round(aspect, 2),
        "centroid_y": round(centroid_ratio, 2),
        "height_ratio": round(height_ratio, 2),
        "hips_y": round(hips_y, 2)
    }
    
    # CRAWLER DETECTION (ALL 4 signals must match)
    crawler_signals = 0
    if aspect > CRAWL_ASPECT_MIN:
        crawler_signals += 1
    if centroid_ratio > CRAWL_CENTROID_MIN:
        crawler_signals += 1
    if height_ratio < CRAWL_HEIGHT_MAX:
        crawler_signals += 1
    if hips_y > CRAWL_HIPS_MIN:
        crawler_signals += 1
    
    metadata["crawler_signals"] = f"{crawler_signals}/4"
    
    if crawler_signals >= 3:  # Need at least 3/4 signals
        logger.log_threat("CRAWLER_DETECTED", person_id, metadata)
        return "CRAWLING", 0.90, metadata
    
    # Everything else is IDLE (including sitting)
    return "IDLE", 0.85, metadata

# ============================================================================
# WEAPON VALIDATION (TEMPORAL PERSISTENCE)
# ============================================================================
def validate_weapon(person_id, weapon_class, frame_no, weapon_bbox, person_pose):
    """
    Requires 5+ detections in recent frames AND hand proximity
    Returns: (is_valid, reason)
    """
    if person_id not in weapon_evidence:
        weapon_evidence[person_id] = {}
    
    weapon_name = model.names[weapon_class]
    
    if weapon_name not in weapon_evidence[person_id]:
        weapon_evidence[person_id][weapon_name] = []
    
    # Add evidence
    weapon_evidence[person_id][weapon_name].append({
        'frame': frame_no,
        'bbox': weapon_bbox
    })
    
    # Keep only recent frames (last 20 frames = ~0.66s at 30fps)
    recent = [e for e in weapon_evidence[person_id][weapon_name] if frame_no - e['frame'] < 20]
    weapon_evidence[person_id][weapon_name] = recent
    
    # Check persistence
    if len(recent) < WEAPON_PERSIST_FRAMES:
        return False, f"INSUFFICIENT_FRAMES_{len(recent)}/{WEAPON_PERSIST_FRAMES}"
    
    # Check hand proximity (if pose data available)
    if person_pose:
        weapon_center = ((weapon_bbox[0] + weapon_bbox[2]) / 2, 
                         (weapon_bbox[1] + weapon_bbox[3]) / 2)
        
        left_wrist = person_pose.get('left_wrist', (9999, 9999))
        right_wrist = person_pose.get('right_wrist', (9999, 9999))
        
        dist_left = np.hypot(weapon_center[0] - left_wrist[0], weapon_center[1] - left_wrist[1])
        dist_right = np.hypot(weapon_center[0] - right_wrist[0], weapon_center[1] - right_wrist[1])
        
        min_dist = min(dist_left, dist_right)
        
        if min_dist > WEAPON_HAND_DIST:
            return False, f"FAR_FROM_HANDS_dist={min_dist:.1f}px"
    
    # VALID WEAPON
    logger.log_threat("WEAPON_VALIDATED", person_id, {
        "weapon": weapon_name,
        "frames": len(recent),
        "persistence": "CONFIRMED"
    })
    return True, "CONFIRMED"

# ============================================================================
# HOSTAGE DETECTION (PROXIMITY + TIMING)
# ============================================================================
def detect_hostage_situation(verified_users, unknown_users):
    """
    Detects Unknown person within 150px of verified user
    Returns: threat_dict or None
    """
    for v_id, v_data in verified_users.items():
        for u_id, u_data in unknown_users.items():
            v_center = bbox_center(v_data['bbox'])
            u_center = bbox_center(u_data['bbox'])
            
            distance = np.hypot(v_center[0] - u_center[0], v_center[1] - u_center[1])
            
            # Check timing (unknown appeared after verified)
            time_gap = u_data['timestamp'] - v_data['timestamp']
            
            # Check if unknown is behind (spatially)
            is_behind = u_center[1] < v_center[1]
            
            if distance < 150 and 0 < time_gap < 5.0 and is_behind:
                evidence = {
                    "verified_user": v_data['identity'],
                    "unknown_id": u_id,
                    "distance_px": round(distance, 1),
                    "time_gap": round(time_gap, 2),
                    "spatial": "BEHIND"
                }
                logger.log_threat("HOSTAGE_PATTERN", u_id, evidence)
                return evidence
    
    return None

# ============================================================================
# RESULT CHECKING (FACE & POSE WORKERS)
# ============================================================================
def check_results():
    global last_verified_time
    
    # FACE RESULTS
    for f in glob.glob("results_face/*.txt"):
        try:
            jid = os.path.basename(f).split('.')[0].replace('result_', '')
            with open(f, 'r') as file:
                res = file.read().strip()
            
            if jid in person_memory:
                mem = person_memory[jid]
                mem['face'] = res
                
                # Anti-spoof handling
                if "ERROR:SPOOF" in res:
                    logger.log_threat("SPOOF_DETECTED", jid, {"reason": res})
                    send_cmd("FIX_BLUR")
                    mem['status'] = 'SUSPECTED_SPOOF'
                
                # Valid identity
                elif res not in ["Unknown", "Verifying...", "ERROR:BLURRY", "ERROR:DARK", "Error"]:
                    last_verified_time = time.time()
                    if mem['status'] != 'VERIFIED':
                        mem['status'] = 'VERIFIED'
                        mem['identity'] = res
                        logger.log_detection("DOOR", jid, "VERIFIED", 0.95, {"identity": res})
                        send_cmd(f"WELCOME:{res}")
                
                elif res == "Unknown":
                    if time.time() - last_verified_time > 15.0:
                        if time.time() - mem['start_time'] > 8.0 and not mem['warned']:
                            logger.log_detection("DOOR", jid, "STRANGER_LINGERING", 0.80, {})
                            mem['warned'] = True
                            send_cmd("ASK_NAME")
                
                elif res == "ERROR:BLURRY":
                    send_cmd("FIX_BLUR")
                elif res == "ERROR:DARK":
                    send_cmd("FIX_DARK")
            
            os.remove(f)
        except Exception as e:
            logger.log_threat("RESULT_PARSE_ERROR", "face", {"error": str(e)})
    
    # POSE RESULTS
    for f in glob.glob("results_pose/*.txt"):
        try:
            jid = os.path.basename(f).split('.')[0].replace('result_', '')
            with open(f, 'r') as file:
                res = file.read().strip()
            
            if jid in person_memory:
                person_memory[jid]['pose_raw'] = res
                
                # Parse: STATUS|speed|hips_y=0.85,history_len=8
                parts = res.split('|')
                status = parts[0] if len(parts) > 0 else 'Normal'
                
                # Extract metadata
                meta = {}
                if len(parts) >= 3:
                    for kv in parts[2].split(','):
                        if '=' in kv:
                            k, v = kv.split('=')
                            try:
                                meta[k.strip()] = float(v)
                            except:
                                pass
                
                person_memory[jid]['pose_hints'] = meta
                
                if "Violent" in status or "Threat" in status:
                    logger.log_threat("VIOLENCE_DETECTED", jid, {"pose": status, "meta": meta})
                    send_cmd("WARN_INTRUDER")
                elif status == "Threat: Surrender":
                    person_memory[jid]['pose'] = 'SURRENDER'
            
            os.remove(f)
        except Exception as e:
            logger.log_threat("RESULT_PARSE_ERROR", "pose", {"error": str(e)})

# ============================================================================
# FRAME DISPATCH
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
    
    cap.release()

# ============================================================================
# INITIALIZATION
# ============================================================================
for d in ["jobs_face", "results_face", "jobs_pose", "results_pose"]:
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*"):
        try:
            os.remove(f)
        except:
            pass

threading.Thread(target=capture, args=(RTSP_URL_GATE, q_gate, "GATE"), daemon=True).start()
threading.Thread(target=capture, args=(RTSP_URL_DOOR, q_door, "DOOR"), daemon=True).start()

print("[SYSTEM] Cameras Active. Entering Main Loop...")
logger.log_detection("SYSTEM", "MAIN", "LOOP_START", 1.0, {})

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
loop_start = time.time()

while True:
    try:
        img_g = q_gate.get(timeout=1)
        img_d = q_door.get(timeout=1)
    except:
        continue
    
    frame_cnt += 1
    iter_start = time.time()
    
    ai_g = cv2.resize(img_g, (AI_WIDTH, AI_HEIGHT))
    ai_d = cv2.resize(img_d, (AI_WIDTH, AI_HEIGHT))
    
    # YOLO inference every 3rd frame (10 FPS)
    if frame_cnt % 3 == 0:
        yolo_start = time.time()
        
        res_g = model(ai_g, classes=[0] + WEAPON_CLASSES + PACKAGE_CLASSES, 
                      verbose=False, conf=WEAPON_CONF)[0]
        res_d = model(ai_d, classes=[0] + WEAPON_CLASSES + PACKAGE_CLASSES, 
                      verbose=False, conf=WEAPON_CONF)[0]
        
        logger.log_perf("YOLO", "inference_ms", (time.time() - yolo_start) * 1000)
        
        # Process detections
        def parse_detections(res, tracker, cam_type, img_src):
            boxes = []
            weapons_detected = []
            packages_detected = []
            
            for box in res.boxes:
                cls = int(box.cls[0])
                coords = box.xyxy[0].cpu().numpy()
                
                if cls == 0:  # Person
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
            
            # Update tracker
            tracks = tracker.update(np.array(boxes) if boxes else np.empty((0, 5)))
            
            for d in tracks:
                x1, y1, x2, y2, tid = map(int, d)
                jid = f"{cam_type}_{tid}"
                
                # Initialize memory
                if jid not in person_memory:
                    person_memory[jid] = {
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
                        'posture': 'IDLE'
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
                
                # POSTURE ANALYSIS
                posture_status, post_conf, post_meta = analyze_posture(
                    jid, [x1, y1, x2, y2], AI_HEIGHT, mem['pose_hints']
                )
                mem['posture'] = posture_status
                
                # THREAT EVALUATION
                # Crawling always triggers alarm
                if posture_status == "CRAWLING":
                    logger.log_threat("CRAWLER_ALARM", jid, post_meta)
                    send_cmd("WARN_INTRUDER")
                
                # Running/Charging (only if not verified or verified but charging)
                if movement_status in ["RUNNING", "CHARGING"]:
                    if mem['identity'] not in AUTHORIZED_USERS or movement_status == "CHARGING":
                        logger.log_threat("RUNNER_ALARM", jid, move_meta)
                        send_cmd("WARN_RUNNING")
                
                # Weapon validation
                for weapon in weapons_detected:
                    is_valid, reason = validate_weapon(
                        jid, weapon['class'], frame_cnt, weapon['bbox'], 
                        mem.get('pose_hints', {})
                    )
                    if is_valid:
                        logger.log_threat("WEAPON_ALARM", jid, {
                            "weapon": weapon['name'],
                            "reason": reason
                        })
                        send_cmd("WARN_WEAPON")
                
                # Log state
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
                
                # Dispatch crops (optimized intervals)
                crop = img_src[max(0, y1):min(AI_HEIGHT, y2), max(0, x1):min(AI_WIDTH, x2)]
                
                if crop.size > 0:
                    # Pose every 2nd frame (15 FPS)
                    if cam_type == 'gate' and frame_cnt % 2 == 0:
                        dispatch('pose', crop, jid)
                    
                    # Face every 5th frame (6 FPS) - only door camera
                    if cam_type == 'door' and frame_cnt % 5 == 0:
                        dispatch('face', crop, jid)
        
        parse_detections(res_g, gate_tracker, 'gate', ai_g)
        parse_detections(res_d, door_tracker, 'door', ai_d)
    
    # Check worker results
    check_results()
    
    # Hostage detection (every 10 frames)
    if frame_cnt % 10 == 0:
        verified = {k: v for k, v in person_memory.items() if v['identity'] in AUTHORIZED_USERS}
        unknown = {k: v for k, v in person_memory.items() if v['identity'] == 'Unknown' and v['face'] == 'Unknown'}
        
        hostage = detect_hostage_situation(verified, unknown)
        if hostage:
            send_cmd("SILENT_ALARM")
    
    # DISPLAY (visual debug)
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
                color = (0, 255, 0)  # Green
            elif posture == "CRAWLING" or movement in ["RUNNING", "CHARGING"]:
                color = (0, 0, 255)  # Red
            elif identity == "Unknown":
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
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
    
    # Performance logging (every 100 frames)
    if frame_cnt % 100 == 0:
        logger.log_perf("MAIN", "loop_fps", fps)
        logger.log_perf("MAIN", "tracked_persons", len(person_memory))
    
    if cv2.waitKey(1) == ord('q'):
        logger.log_detection("SYSTEM", "MAIN", "SHUTDOWN", 1.0, {})
        break

cv2.destroyAllWindows()