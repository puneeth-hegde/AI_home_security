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

# --- Configuration ---
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
YOLO_MODEL_PATH = "yolov8n.pt"
TARGET_DISPLAY_HEIGHT = 720
TARGET_DISPLAY_WIDTH = 1280

# --- VRAM FIX: Inference Resolution ---
AI_WIDTH = 640
AI_HEIGHT = 360

# --- Optimization ---
PROCESS_EVERY_N_FRAMES = 5

# --- Specialist Worker Job Folders ---
JOBS_FACE_DIR = "jobs_face"
RESULTS_FACE_DIR = "results_face"
JOBS_POSE_DIR = "jobs_pose"
RESULTS_POSE_DIR = "results_pose"

# --- Startup Diagnostics ---
print("--- MAIN APPLICATION STARTING (BRAIN_ENV) ---")
gpu_available = torch.cuda.is_available()
device = 0 if gpu_available else 'cpu'
print(f"[{datetime.now()}] [SETUP] GPU Available: {gpu_available}")
print(f"[{datetime.now()}] [SETUP] Using Device: {device}")

# Load model
model = YOLO(YOLO_MODEL_PATH)
model.to(device)
print("[INFO] YOLOv8 model loaded.")

gate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
door_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

gate_frame_queue = queue.Queue(maxsize=2)
door_frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()
world_state = {}
original_frame_shapes = {"GATE": None, "DOOR": None}

# --- Frame Capture Function ---
def capture_frames(rtsp_url, frame_queue, camera_name):
    global original_frame_shapes
    print(f"[{datetime.now()}] [CAPTURE-{camera_name}] Connecting to stream...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not video_capture.isOpened():
        print(f"[{datetime.now()}] [ERROR-{camera_name}] Could not open video stream.")
        stop_event.set()
        return
    print(f"[{datetime.now()}] [CAPTURE-{camera_name}] Connection successful!")
    
    first_frame = True
    while not stop_event.is_set():
        ret, frame = video_capture.read()
        if not ret:
            time.sleep(0.5)
            continue
        if first_frame:
            original_frame_shapes[camera_name] = frame.shape
            print(f"[{datetime.now()}] [DIAGNOSTIC-{camera_name}] Original Frame Shape: {frame.shape}")
            first_frame = False
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(frame)
        time.sleep(0.01)
    video_capture.release()
    print(f"[{datetime.now()}] [CAPTURE-{camera_name}] Thread stopped.")

# --- Specialist Job Dispatcher Function ---
def dispatch_job(job_env, worker_script, job_image_crop, job_id):
    try:
        job_dir = JOBS_FACE_DIR if job_env == 'face_env' else JOBS_POSE_DIR
        job_image_path = os.path.join(job_dir, f"{job_id}.jpg")
        
        # FIX: ONLY Save the image. Do NOT spawn a new process.
        # The external worker terminals are already running and watching these folders.
        cv2.imwrite(job_image_path, job_image_crop)
        
        print(f"[{datetime.now()}] [BRAIN] Dispatched Job {job_id} to {job_env}")
    except Exception as e:
        print(f"[{datetime.now()}] [ERROR] Failed to dispatch job {job_id}: {e}")

# --- Specialist Result Checker Function ---
def check_for_results():
    global world_state
    
    # Check Face Results
    for result_file in glob.glob(os.path.join(RESULTS_FACE_DIR, "*.txt")):
        try:
            filename = os.path.basename(result_file)
            job_id = filename.replace('result_', '').replace('.txt', '')
            
            with open(result_file, 'r') as f:
                identity = f.read().strip()
            
            if identity and identity != "Error":
                if job_id in world_state:
                    if world_state[job_id]['identity'] == "Pending...":
                        print(f"[{datetime.now()}] [BRAIN] Received Result for {job_id}: Identity is {identity}")
                    world_state[job_id]['identity'] = identity
            os.remove(result_file)
        except (IOError, PermissionError):
            pass 
        except Exception as e:
            print(f"[{datetime.now()}] [ERROR] Could not read face result file {result_file}: {e}")

    # Check Pose Results
    for result_file in glob.glob(os.path.join(RESULTS_POSE_DIR, "*.txt")):
        try:
            filename = os.path.basename(result_file)
            job_id = filename.replace('result_', '').replace('.txt', '')
            
            with open(result_file, 'r') as f:
                pose_status = f.read().strip()
            
            if pose_status and pose_status != "Error":
                if job_id in world_state:
                    if world_state[job_id]['pose'] == "Pending...":
                          print(f"[{datetime.now()}] [BRAIN] Received Result for {job_id}: Pose is {pose_status}")
                    world_state[job_id]['pose'] = pose_status
            os.remove(result_file)
        except (IOError, PermissionError):
            pass
        except Exception as e:
            print(f"[{datetime.now()}] [ERROR] Could not read pose result file {result_file}: {e}")

# --- Start Background Services ---
print(f"[{datetime.now()}] [SETUP] Cleaning up old job/result files...")
for f in glob.glob(os.path.join(JOBS_FACE_DIR, "*.*")): os.remove(f)
for f in glob.glob(os.path.join(RESULTS_FACE_DIR, "*.*")): os.remove(f)
for f in glob.glob(os.path.join(JOBS_POSE_DIR, "*.*")): os.remove(f)
for f in glob.glob(os.path.join(RESULTS_POSE_DIR, "*.*")): os.remove(f)

gate_thread = threading.Thread(target=capture_frames, args=(RTSP_URL_GATE, gate_frame_queue, "GATE"), daemon=True)
door_thread = threading.Thread(target=capture_frames, args=(RTSP_URL_DOOR, door_frame_queue, "DOOR"), daemon=True)
gate_thread.start()
door_thread.start()

print(f"[{datetime.now()}] [INFO] All services started. Starting main loop...")
time.sleep(2) 

# --- Main Application Loop ---
prev_frame_time = time.time()
frame_counter = 0
tracked_objects_gate = np.empty((0, 5))
tracked_objects_door = np.empty((0, 5))
active_job_ids = set() 

while not stop_event.is_set():
    gate_frame = None
    door_frame = None
    try: gate_frame = gate_frame_queue.get(timeout=1)
    except queue.Empty: pass
    try: door_frame = door_frame_queue.get(timeout=1)
    except queue.Empty: pass

    if gate_frame is None or door_frame is None or original_frame_shapes["GATE"] is None or original_frame_shapes["DOOR"] is None:
        if not gate_thread.is_alive() or not door_thread.is_alive():
            stop_event.set()
        continue

    frame_counter += 1
    
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        # --- VRAM FIX: Resize on CPU before sending to GPU ---
        orig_h, orig_w = gate_frame.shape[:2]
        scale_x = orig_w / AI_WIDTH
        scale_y = orig_h / AI_HEIGHT
        
        gate_small = cv2.resize(gate_frame, (AI_WIDTH, AI_HEIGHT))
        door_small = cv2.resize(door_frame, (AI_WIDTH, AI_HEIGHT))
        
        results_gate = model(gate_small, classes=[0], verbose=False)
        results_door = model(door_small, classes=[0], verbose=False)
        
        detections_gate = np.empty((0, 4))
        if results_gate and results_gate[0].boxes.xyxy.numel() > 0:
            boxes = results_gate[0].boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            detections_gate = boxes

        detections_door = np.empty((0, 4))
        if results_door and results_door[0].boxes.xyxy.numel() > 0:
            boxes = results_door[0].boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            detections_door = boxes
            
        tracked_objects_gate = gate_tracker.update(detections_gate)
        tracked_objects_door = door_tracker.update(detections_door)

        current_time = time.time()
        active_job_ids = set() 

        # --- Brain Logic: Gate ---
        for d in tracked_objects_gate:
            x1, y1, x2, y2, track_id = map(int, d)
            job_id = f"gate_{track_id}"
            active_job_ids.add(job_id)
            
            if job_id not in world_state:
                print(f"[{datetime.now()}] [BRAIN] New person {job_id} detected at Gate.")
                world_state[job_id] = {'identity': 'N/A', 'pose': 'Pending...', 'last_seen': current_time, 'face_job_sent': True, 'pose_job_sent': False}

            world_state[job_id]['last_seen'] = current_time
            world_state[job_id]['box'] = (x1, y1, x2, y2)
            
            if not world_state[job_id]['pose_job_sent']:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                crop = gate_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    dispatch_job('pose_env', 'pose_worker.py', crop, job_id)
                    world_state[job_id]['pose_job_sent'] = True
        
        # --- Brain Logic: Door ---
        for d in tracked_objects_door:
            x1, y1, x2, y2, track_id = map(int, d)
            job_id = f"door_{track_id}"
            active_job_ids.add(job_id)
            
            if job_id not in world_state:
                print(f"[{datetime.now()}] [BRAIN] New person {job_id} detected at Door.")
                world_state[job_id] = {'identity': 'Pending...', 'pose': 'Pending...', 'last_seen': current_time, 'face_job_sent': False, 'pose_job_sent': False}
            
            world_state[job_id]['last_seen'] = current_time
            world_state[job_id]['box'] = (x1, y1, x2, y2) 
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            if not world_state[job_id]['face_job_sent']:
                crop = door_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    dispatch_job('face_env', 'face_worker.py', crop, job_id)
                    world_state[job_id]['face_job_sent'] = True
            
            if (world_state[job_id]['identity'] == 'Unknown' and 
                not world_state[job_id]['pose_job_sent']):
                crop = door_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    dispatch_job('pose_env', 'pose_worker.py', crop, job_id)
                    world_state[job_id]['pose_job_sent'] = True
    
    check_for_results()

    try:
        gate_frame_display = cv2.resize(gate_frame, (TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT))
        door_frame_display = cv2.resize(door_frame, (TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT))

        orig_h_gate, orig_w_gate, _ = original_frame_shapes["GATE"]
        scale_x_gate = TARGET_DISPLAY_WIDTH / orig_w_gate
        scale_y_gate = TARGET_DISPLAY_HEIGHT / orig_h_gate

        orig_h_door, orig_w_door, _ = original_frame_shapes["DOOR"]
        scale_x_door = TARGET_DISPLAY_WIDTH / orig_w_door
        scale_y_door = TARGET_DISPLAY_HEIGHT / orig_h_gate

        global_threat_detected = False 

        # --- Drawing Loop: Gate ---
        for d in tracked_objects_gate:
            x1, y1, x2, y2, track_id = map(int, d)
            job_id = f"gate_{track_id}"
            state = world_state.get(job_id)
            if state:
                pose_status = state.get('pose', '...')
                if "Threat" in pose_status: global_threat_detected = True 

                x1_d, y1_d = int(x1 * scale_x_gate), int(y1 * scale_y_gate)
                x2_d, y2_d = int(x2 * scale_x_gate), int(y2 * scale_y_gate)
                cv2.rectangle(gate_frame_display, (x1_d, y1_d), (x2_d, y2_d), (255, 0, 0), 2)
                cv2.putText(gate_frame_display, f"Person {track_id} (Pose: {pose_status})", (x1_d, y1_d - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- Drawing Loop: Door ---
        for d in tracked_objects_door:
            x1, y1, x2, y2, track_id = map(int, d)
            job_id = f"door_{track_id}"
            state = world_state.get(job_id)
            if state:
                identity = state.get('identity', 'Processing...')
                pose_status = state.get('pose', '...')
                if "Threat" in pose_status: global_threat_detected = True
                
                color = (0, 255, 0) if identity not in ['Unknown', 'Pending...', 'Processing...'] else (0, 0, 255)
                x1_d, y1_d = int(x1 * scale_x_door), int(y1 * scale_y_door)
                x2_d, y2_d = int(x2 * scale_x_door), int(y2 * scale_y_door)

                cv2.rectangle(door_frame_display, (x1_d, y1_d), (x2_d, y2_d), color, 2)
                text = f"{identity} (ID: {track_id})"
                text2 = f"Pose: {pose_status}"
                cv2.rectangle(door_frame_display, (x1_d, y1_d - 40), (x1_d + 250, y1_d), color, cv2.FILLED)
                cv2.putText(door_frame_display, text, (x1_d + 6, y1_d - 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(door_frame_display, text2, (x1_d + 6, y1_d - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # --- Clean up old state data ---
        if frame_counter % 30 == 0: 
            all_known_job_ids = list(world_state.keys())
            for job_id in all_known_job_ids:
                if job_id not in active_job_ids: 
                    print(f"[{datetime.now()}] [BRAIN] Person {job_id} is no longer tracked. Removing state.")
                    del world_state[job_id]

        combined_frame = cv2.hconcat([gate_frame_display, door_frame_display])

        # --- Draw Global Threat Alert ---
        if global_threat_detected:
            cv2.putText(combined_frame, "THREAT DETECTED!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
        
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(combined_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, f"GPU Active: {gpu_available}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('AI Security System - Main App', combined_frame)

    except Exception as e:
        print(f"[{datetime.now()}] [ERROR] Frame drawing error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

# --- Cleanup ---
print(f"[{datetime.now()}] [INFO] Cleaning up...")
stop_event.set()
gate_thread.join()
door_thread.join()
cv2.destroyAllWindows()
print(f"[{datetime.now()}] [INFO] Application closed.")