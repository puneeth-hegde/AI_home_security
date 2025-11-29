import os
import time
import cv2
import glob
import math
import mediapipe as mp
from collections import deque
from datetime import datetime

# --- CONFIGURATION ---
JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"

# --- SENSITIVITY ---
# 0.02 = Sensitive (Detects shoves/fast walking)
# 0.06 = Heavy (Detects hard punches)
VIOLENCE_SPEED_THRESHOLD = 0.02 
HISTORY_SIZE = 5

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
tracking_memory = {}

# Create Dirs
for d in [JOBS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE] SENSITIVE ENGINE READY.")

def calculate_distance(point_a, point_b):
    return math.sqrt((point_a.x - point_b.x)**2 + (point_a.y - point_b.y)**2)

def analyze_violence(landmarks, job_id):
    # 1. Get Key Points
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # 2. Check Surrender (Wrists above Nose)
    # Note: Y is inverted (0 is top)
    if (left_wrist.y < nose.y) and (right_wrist.y < nose.y):
        return "Threat: Surrender Pose", 0.0

    # 3. Check Velocity
    current_data = {
        'l': (left_wrist.x, left_wrist.y), 
        'r': (right_wrist.x, right_wrist.y), 
        't': time.time()
    }
    
    if job_id not in tracking_memory: 
        tracking_memory[job_id] = deque(maxlen=HISTORY_SIZE)
    tracking_memory[job_id].append(current_data)
    
    speed = 0.0
    status = "Normal"

    if len(tracking_memory[job_id]) > 2:
        # Compare oldest frame in memory to current frame
        start = tracking_memory[job_id][0]
        end = tracking_memory[job_id][-1]
        dt = end['t'] - start['t']
        
        if dt > 0 and dt < 1.0:
            l_dist = math.sqrt((start['l'][0]-end['l'][0])**2 + (start['l'][1]-end['l'][1])**2)
            r_dist = math.sqrt((start['r'][0]-end['r'][0])**2 + (start['r'][1]-end['r'][1])**2)
            
            speed = max(l_dist, r_dist) / dt
            
            if speed > VIOLENCE_SPEED_THRESHOLD:
                status = "Threat: Violent Motion"

    return status, speed

while True:
    try:
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        if not job_files:
            time.sleep(0.05)
            continue

        for job_path in job_files:
            job_filename = os.path.basename(job_path)
            job_id = job_filename.split('.')[0] 
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            
            try:
                image = cv2.imread(job_path)
                if image is None: raise Exception("Empty")
                
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                status = "Normal"
                speed = 0.0
                
                if results.pose_landmarks:
                    status, speed = analyze_violence(results.pose_landmarks.landmark, job_id)
                    
                    # VERBOSE LOGGING
                    if status != "Normal":
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE] ALERT: {status} on {job_id} (Speed: {speed:.3f})")
                    # else:
                        # Uncomment to see normal speeds for tuning
                        # print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE] {job_id} Speed: {speed:.3f}")
                
                with open(result_path, "w") as f: f.write(status)

            except:
                with open(result_path, "w") as f: f.write("Error")
            
            # Cleanup
            try: os.remove(job_path)
            except: pass

    except KeyboardInterrupt: break
    except Exception: time.sleep(0.1)