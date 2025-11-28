import cv2
import os
import time
from datetime import datetime
import threading
import queue
import numpy as np

# --- Configuration ---
# --- IMPORTANT: Replace these with your actual RTSP URLs ---
# --- Make sure BOTH are using stream1 for a real test ---
RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"

# --- We will FORCE both frames to this size ---
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# --- Thread-safe Queues & Stop Event ---
gate_frame_queue = queue.Queue(maxsize=2)
door_frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()
gate_read_errors = 0
door_read_errors = 0

# --- Frame Capture Function ---
def capture_frames(rtsp_url, frame_queue, camera_name):
    global gate_read_errors, door_read_errors
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
            print(f"[{datetime.now()}] [ERROR-{camera_name}] Failed to read frame.")
            if camera_name == "GATE": gate_read_errors += 1
            else: door_read_errors += 1
            time.sleep(0.5)
            continue

        if first_frame:
            print(f"[{datetime.now()}] [DIAGNOSTIC-{camera_name}] Original Frame Shape: {frame.shape}")
            first_frame = False

        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        
        frame_queue.put(frame)
        time.sleep(0.01)

    video_capture.release()
    print(f"[{datetime.now()}] [CAPTURE-{camera_name}] Thread stopped.")

# --- Start Capture Threads ---
gate_thread = threading.Thread(target=capture_frames, args=(RTSP_URL_GATE, gate_frame_queue, "GATE"), daemon=True)
door_thread = threading.Thread(target=capture_frames, args=(RTSP_URL_DOOR, door_frame_queue, "DOOR"), daemon=True)
gate_thread.start()
door_thread.start()

# --- Main Display Loop ---
print(f"[{datetime.now()}] [INFO] Starting main display loop...")
prev_frame_time = time.time()

while not stop_event.is_set():
    gate_frame = None
    door_frame = None
    try: gate_frame = gate_frame_queue.get(timeout=1)
    except queue.Empty: pass
    try: door_frame = door_frame_queue.get(timeout=1)
    except queue.Empty: pass

    if gate_frame is None or door_frame is None:
        if not gate_thread.is_alive() or not door_thread.is_alive():
            print(f"[{datetime.now()}] [ERROR] A camera thread has stopped. Exiting.")
            stop_event.set()
        continue

    # --- FIXED DISPLAY LOGIC ---
    try:
        # Force Gate frame to target size
        gate_frame_resized = cv2.resize(gate_frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Force Door frame to target size
        door_frame_resized = cv2.resize(door_frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # Combine the resized frames
        combined_frame = cv2.hconcat([gate_frame_resized, door_frame_resized])

    except Exception as e:
        print(f"[{datetime.now()}] [ERROR] Frame processing error: {e}")
        continue

    # --- Performance Calculation & Display ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    
    cv2.putText(combined_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_frame, "GATE (Errors: " + str(gate_read_errors) + ")", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    door_text_x = gate_frame_resized.shape[1] + 10 
    cv2.putText(combined_frame, "DOOR (Errors: " + str(door_read_errors) + ")", (door_text_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Dual Camera Network Test (FIXED)', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

# --- Cleanup ---
print(f"[{datetime.now()}] [INFO] Cleaning up...")
gate_thread.join()
door_thread.join()
cv2.destroyAllWindows()
print(f"[{datetime.now()}] [INFO] Application closed.")