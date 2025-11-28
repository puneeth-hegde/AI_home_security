import os
import time
import cv2
import glob
import shutil
import math
import mediapipe as mp
from datetime import datetime

# --- CONFIGURATION ---
JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"
PROCESSING_DIR = "jobs_pose_processing"
POLLING_INTERVAL = 0.05

# --- SENSITIVITY ---
# 0.02 is EXTREMELY sensitive. Even a fast wave will trigger it.
VIOLENCE_SPEED_THRESHOLD = 0.02 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
tracking_memory = {}

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE_WORKER] Violence Engine Ready.")

def calculate_distance(point_a, point_b):
    return math.sqrt((point_a.x - point_b.x)**2 + (point_a.y - point_b.y)**2)

def analyze_violence(landmarks, job_id):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # 1. Surrender Check
    if (left_wrist.y < nose.y) and (right_wrist.y < nose.y):
        return "Threat: Surrender Pose"

    # 2. Velocity Check
    if job_id in tracking_memory:
        prev_data = tracking_memory[job_id]
        if time.time() - prev_data['time'] < 1.0: 
            prev = prev_data['landmarks']
            speed_l = calculate_distance(left_wrist, prev[mp_pose.PoseLandmark.LEFT_WRIST])
            speed_r = calculate_distance(right_wrist, prev[mp_pose.PoseLandmark.RIGHT_WRIST])
            
            # DEBUG PRINT: Verify math is happening
            # print(f"DEBUG: {job_id} Speed L:{speed_l:.4f} R:{speed_r:.4f}")

            if speed_l > VIOLENCE_SPEED_THRESHOLD or speed_r > VIOLENCE_SPEED_THRESHOLD:
                return "Threat: Violent Motion"
            
    tracking_memory[job_id] = {'landmarks': landmarks, 'time': time.time()}
    return "Normal"

while True:
    try:
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        if not job_files:
            time.sleep(POLLING_INTERVAL)
            continue

        for job_path in job_files:
            job_filename = os.path.basename(job_path)
            job_id = job_filename.split('.')[0] 
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            processing_path = os.path.join(PROCESSING_DIR, job_filename)
            
            try:
                shutil.move(job_path, processing_path)
            except: continue

            try:
                image = cv2.imread(processing_path)
                if image is None: raise Exception("Empty")
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                status = "Normal"
                
                if results.pose_landmarks:
                    status = analyze_violence(results.pose_landmarks.landmark, job_id)
                    if "Threat" in status:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ALERT] {status} on {job_id}!")
                
                with open(result_path, "w") as f: f.write(status)
            except:
                with open(result_path, "w") as f: f.write("Error")
            
            if os.path.exists(processing_path): os.remove(processing_path)

    except KeyboardInterrupt: break
    except Exception: time.sleep(0.1)