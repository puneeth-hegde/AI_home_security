import os
import time
import cv2
import glob
import math
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime

# --- CONFIGURATION ---
JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"
PROCESSING_DIR = "jobs_pose_processing"

# --- INTELLIGENT THRESHOLDS ---
# 0.02 is sensitive enough for a shove, but ignores walking
VIOLENCE_SPEED_THRESHOLD = 0.02 
# How many frames to remember to calculate speed?
HISTORY_SIZE = 5 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# Stores history of movements for every person ID
tracking_memory = {}

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE_WORKER] Hybrid Intelligence Engine Ready.")

def calculate_distance(point_a, point_b):
    """Simple Euclidean distance"""
    return math.sqrt((point_a.x - point_b.x)**2 + (point_a.y - point_b.y)**2)

def analyze_violence(landmarks, job_id):
    # 1. Get Key Body Parts
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # 2. Check Surrender (Geometry: Wrists above Nose)
    # In images, Y=0 is top. So Wrist.y < Nose.y means Wrist is ABOVE Nose.
    if (left_wrist.y < nose.y) and (right_wrist.y < nose.y):
        return "Threat: Surrender Pose"

    # 3. Check Velocity (Physics: Distance / Time)
    current_data = {
        'left': (left_wrist.x, left_wrist.y), 
        'right': (right_wrist.x, right_wrist.y), 
        'time': time.time()
    }
    
    # Initialize memory for new person
    if job_id not in tracking_memory: 
        tracking_memory[job_id] = deque(maxlen=HISTORY_SIZE)
    
    tracking_memory[job_id].append(current_data)
    
    # Need at least 2 frames to calculate speed
    if len(tracking_memory[job_id]) < 2: 
        return "Normal"
    
    # Compare "Now" vs "Oldest Frame in Memory" (Smoother than frame-by-frame)
    start = tracking_memory[job_id][0]
    end = tracking_memory[job_id][-1]
    dt = end['time'] - start['time']
    
    # If the time gap is valid (not 0, and not stale > 1 second)
    if dt > 0 and dt < 1.0:
        # Calculate how far hands moved
        l_dist = math.sqrt((start['left'][0]-end['left'][0])**2 + (start['left'][1]-end['left'][1])**2)
        r_dist = math.sqrt((start['right'][0]-end['right'][0])**2 + (start['right'][1]-end['right'][1])**2)
        
        # Calculate Speed (Distance / Time)
        speed = max(l_dist, r_dist) / dt
        
        # DEBUG: Uncomment this if you want to see the numbers
        # print(f"[{job_id}] Speed: {speed:.4f}")
        
        if speed > VIOLENCE_SPEED_THRESHOLD:
            return "Threat: Violent Motion"

    return "Normal"

# --- CLEANUP LOOP ---
# Prevents memory leak by removing old IDs
def clean_memory():
    now = time.time()
    for jid in list(tracking_memory.keys()):
        if now - tracking_memory[jid][-1]['time'] > 5.0: # Forget after 5 secs
            del tracking_memory[jid]

last_clean = time.time()

while True:
    try:
        # Periodic Memory Cleanup (Every 5 seconds)
        if time.time() - last_clean > 5.0:
            clean_memory()
            last_clean = time.time()

        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        if not job_files:
            time.sleep(0.05)
            continue

        for job_path in job_files:
            job_filename = os.path.basename(job_path)
            job_id = job_filename.split('.')[0] 
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            processing_path = os.path.join(PROCESSING_DIR, job_filename)
            
            # Atomic Move (Prevents reading half-written files)
            try:
                shutil.move(job_path, processing_path)
            except: continue

            try:
                image = cv2.imread(processing_path)
                if image is None: raise Exception("Empty")
                
                # Run MediaPipe
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                status = "Normal"
                if results.pose_landmarks:
                    status = analyze_violence(results.pose_landmarks.landmark, job_id)
                    
                    if "Threat" in status:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ALERT] {status} on {job_id}")
                
                with open(result_path, "w") as f: f.write(status)
            except:
                with open(result_path, "w") as f: f.write("Error")
            
            if os.path.exists(processing_path): os.remove(processing_path)

    except KeyboardInterrupt: break
    except Exception as e: 
        print(f"Critical Error: {e}")
        time.sleep(0.1)