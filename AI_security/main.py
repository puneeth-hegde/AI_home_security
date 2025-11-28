import cv2
import mediapipe as mp
from ultralytics import YOLO
import os
import time
import torch
from datetime import datetime
import threading
import queue
import face_recognition # For face recognition
import pickle # For loading encodings
import numpy as np # For resizing frames

# --- Configuration ---
# --- IMPORTANT: Replace these with your actual RTSP URLs ---
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"

YOLO_MODEL_PATH = "yolov8n.pt"
ENCODINGS_FILE = "encodings.pkl"

# --- Optimization ---
PROCESS_EVERY_N_FRAMES = 10 # Run heavy AI only every few frames

# --- Startup Diagnostics ---
print("--- MAIN APPLICATION STARTING ---")
gpu_available = torch.cuda.is_available()
device = 'cuda' if gpu_available else 'cpu'
print(f"[{datetime.now()}] [SETUP] GPU Available: {gpu_available}")
print(f"[{datetime.now()}] [SETUP] Using Device: {device.upper()}")

# --- Load Models ---
print(f"[{datetime.now()}] [SETUP] Loading YOLOv8 model: {YOLO_MODEL_PATH}")
model = YOLO(YOLO_MODEL_PATH)
model.to(device)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
print(f"[{datetime.now()}] [SETUP] Models loaded successfully.")

# --- Load Face Encodings ---
print(f"[{datetime.now()}] [SETUP] Loading face encodings from: {ENCODINGS_FILE}")
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"[{datetime.now()}] [SETUP] Loaded {len(data['names'])} known faces.")
else:
    data = {"encodings": [], "names": []}
    print(f"[{datetime.now()}] [WARNING] Encodings file not found. Starting with empty face database.")

# --- Thread-safe Queues & Stop Event ---
gate_frame_queue = queue.Queue(maxsize=1)
door_frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

# --- Frame Capture Function (same as before) ---
def capture_frames(rtsp_url, frame_queue, camera_name):
    print(f"[{datetime.now()}] [CAPTURE-{camera_name}] Connecting to stream...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not video_capture.isOpened():
        print(f"[{datetime.now()}] [ERROR-{camera_name}] Could not open video stream.")
        stop_event.set()
        return
    print(f"[{datetime.now()}] [CAPTURE-{camera_name}] Connection successful!")
    while not stop_event.is_set():
        ret, frame = video_capture.read()
        if not ret: continue
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(frame)
    video_capture.release()
    print(f"[{datetime.now()}] [CAPTURE-{camera_name}] Thread stopped.")

# --- Start Capture Threads ---
gate_thread = threading.Thread(target=capture_frames, args=(RTSP_URL_GATE, gate_frame_queue, "GATE"), daemon=True)
door_thread = threading.Thread(target=capture_frames, args=(RTSP_URL_DOOR, door_frame_queue, "DOOR"), daemon=True)
gate_thread.start()
door_thread.start()

# --- Main Processing Loop ---
print(f"[{datetime.now()}] [INFO] Starting main processing loop...")
frame_counter = 0
prev_frame_time = time.time()
last_gate_detections = []
last_door_detections = []

while not stop_event.is_set():
    gate_frame = None
    door_frame = None
    try: gate_frame = gate_frame_queue.get_nowait()
    except queue.Empty: pass
    try: door_frame = door_frame_queue.get_nowait()
    except queue.Empty: pass

    if gate_frame is None or door_frame is None:
        time.sleep(0.01)
        continue

    frame_counter += 1
    current_gate_detections = []
    current_door_detections = []

    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        # --- Process Gate Camera ---
        results_gate = model(gate_frame, classes=[0], verbose=False)
        for result in results_gate:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_roi_rgb = cv2.cvtColor(gate_frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                pose_result = None
                if person_roi_rgb.size > 0:
                    pose_result = pose.process(person_roi_rgb)
                current_gate_detections.append({'box': (x1, y1, x2, y2), 'pose': pose_result})

        # --- Process Door Camera ---
        results_door = model(door_frame, classes=[0], verbose=False)
        small_door_frame = cv2.resize(door_frame, (0, 0), fx=0.5, fy=0.5)
        # --- THIS LINE IS NOW CORRECTED ---
        rgb_door_frame_small = cv2.cvtColor(small_door_frame, cv2.COLOR_BGR2RGB)

        face_locations_small = face_recognition.face_locations(rgb_door_frame_small, model='hog')
        face_encodings_small = face_recognition.face_encodings(rgb_door_frame_small, face_locations_small)

        yolo_boxes_door = []
        if results_door and len(results_door) > 0:
             yolo_boxes_door = results_door[0].boxes.xyxy.cpu().numpy().astype(int)

        for i, (x1_yolo, y1_yolo, x2_yolo, y2_yolo) in enumerate(yolo_boxes_door):
            person_name = "Unknown"
            person_pose = None
            best_face_match_idx = -1
            min_dist = float('inf')

            yolo_center_x = (x1_yolo + x2_yolo) // 2
            yolo_center_y = (y1_yolo + y2_yolo) // 2

            for j, (top, right, bottom, left) in enumerate(face_locations_small):
                 top*=2; right*=2; bottom*=2; left*=2
                 face_center_x = (left + right) // 2
                 face_center_y = (top + bottom) // 2
                 dist = np.sqrt((yolo_center_x - face_center_x)**2 + (yolo_center_y - face_center_y)**2)
                 if (x1_yolo < face_center_x < x2_yolo and y1_yolo < face_center_y < y2_yolo and dist < min_dist):
                     min_dist = dist
                     best_face_match_idx = j

            if best_face_match_idx != -1:
                matches = face_recognition.compare_faces(data["encodings"], face_encodings_small[best_face_match_idx])
                if True in matches:
                    first_match_index = matches.index(True)
                    person_name = data["names"][first_match_index]
                    print(f"[{datetime.now()}] [DOOR-EVENT] Recognized {person_name}")
                else:
                    print(f"[{datetime.now()}] [DOOR-EVENT] Detected Unknown face")

            if person_name == "Unknown":
                person_roi_rgb = cv2.cvtColor(door_frame[y1_yolo:y2_yolo, x1_yolo:x2_yolo], cv2.COLOR_BGR2RGB)
                if person_roi_rgb.size > 0:
                    person_pose = pose.process(person_roi_rgb)

            current_door_detections.append({'box': (x1_yolo, y1_yolo, x2_yolo, y2_yolo), 'name': person_name, 'pose': person_pose})

        last_gate_detections = current_gate_detections
        last_door_detections = current_door_detections

    # --- Drawing runs on EVERY frame using last known results ---
    # Draw on Gate Frame
    for detection in last_gate_detections:
        x1, y1, x2, y2 = detection['box']
        cv2.rectangle(gate_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(gate_frame, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if detection['pose'] and detection['pose'].pose_landmarks:
             mp_drawing.draw_landmarks(gate_frame[y1:y2, x1:x2], detection['pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Draw on Door Frame
    for detection in last_door_detections:
        x1, y1, x2, y2 = detection['box']
        name = detection['name']
        pose_result = detection['pose']
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(door_frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(door_frame, (x1, y1 - 25), (x1 + len(name)*15 , y1), color, cv2.FILLED)
        cv2.putText(door_frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        if pose_result and pose_result.pose_landmarks:
             mp_drawing.draw_landmarks(door_frame[y1:y2, x1:x2], pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # --- Combine and Display ---
    h_gate, w_gate = gate_frame.shape[:2]
    h_door, w_door = door_frame.shape[:2]
    target_height = min(h_gate, h_door, 720)
    if h_gate != target_height:
        scale = target_height / h_gate
        gate_frame = cv2.resize(gate_frame, (int(w_gate * scale), target_height))
    if h_door != target_height:
        scale = target_height / h_door
        door_frame = cv2.resize(door_frame, (int(w_door * scale), target_height))

    combined_frame = cv2.hconcat([gate_frame, door_frame])

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    cv2.putText(combined_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_frame, f"GPU Active: {gpu_available}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('AI Security System - Main App', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

# --- Cleanup ---
print(f"[{datetime.now()}] [INFO] Cleaning up...")
gate_thread.join()
door_thread.join()
cv2.destroyAllWindows()
print(f"[{datetime.now()}] [INFO] Application closed.")