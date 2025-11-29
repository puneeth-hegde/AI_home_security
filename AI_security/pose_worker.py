import os
import time
import cv2
import math
import glob
import numpy as np
import mediapipe as mp
from collections import deque

JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"

VIOLENCE_SPEED_THRESHOLD = 0.04
HISTORY_SIZE = 5

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

print("[POSE] MediaPipe ready")

tracking_memory = {}

class PoseHistory:
    def __init__(self):
        self.wrist_history = deque(maxlen=HISTORY_SIZE)
        self.elbow_history = deque(maxlen=HISTORY_SIZE)
        self.shoulder_history = deque(maxlen=HISTORY_SIZE)
    
    def add_frame(self, wrists, elbows, shoulders):
        self.wrist_history.append(wrists)
        self.elbow_history.append(elbows)
        self.shoulder_history.append(shoulders)
    
    def is_full(self):
        return len(self.wrist_history) >= HISTORY_SIZE

def calculate_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def check_surrender(landmarks):
    try:
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        left_raised = left_wrist.y < left_shoulder.y
        right_raised = right_wrist.y < right_shoulder.y
        
        left_angle = calculate_angle((left_wrist.x, left_wrist.y), (left_elbow.x, left_elbow.y), (left_shoulder.x, left_shoulder.y))
        right_angle = calculate_angle((right_wrist.x, right_wrist.y), (right_elbow.x, right_elbow.y), (right_shoulder.x, right_shoulder.y))
        
        if left_raised and right_raised:
            if left_angle < 100 and right_angle < 100:
                return True
        return False
    except:
        return False

def check_violence(track_id, landmarks):
    if track_id not in tracking_memory:
        tracking_memory[track_id] = PoseHistory()
    
    history = tracking_memory[track_id]
    
    try:
        left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
        right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
        left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
        right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        
        history.add_frame((left_wrist, right_wrist), (left_elbow, right_elbow), (left_shoulder, right_shoulder))
        
        if not history.is_full():
            return False
        
        old_wrists = history.wrist_history[0]
        new_wrists = history.wrist_history[-1]
        
        left_movement = math.sqrt((new_wrists[0][0] - old_wrists[0][0])**2 + (new_wrists[0][1] - old_wrists[0][1])**2)
        right_movement = math.sqrt((new_wrists[1][0] - old_wrists[1][0])**2 + (new_wrists[1][1] - old_wrists[1][1])**2)
        
        max_movement = max(left_movement, right_movement)
        
        if max_movement > VIOLENCE_SPEED_THRESHOLD:
            return True
        return False
    except:
        return False

def analyze_pose(track_id, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        return "NORMAL"
    
    landmarks = results.pose_landmarks.landmark
    
    if check_surrender(landmarks):
        return "SURRENDER"
    
    if check_violence(track_id, landmarks):
        return "VIOLENCE"
    
    return "NORMAL"

print("[POSE] Worker started")

while True:
    job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
    if len(job_files) == 0:
        time.sleep(0.1)
        continue
    
    job_path = sorted(job_files)[0]
    
    try:
        filename = os.path.basename(job_path)
        track_id = int(filename.split('_')[0])
        img = cv2.imread(job_path)
        
        if img is None:
            os.remove(job_path)
            continue
        
        pose_status = analyze_pose(track_id, img)
        
        result_filename = f"{track_id}_{int(time.time()*1000)}.txt"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        with open(result_path, 'w') as f:
            f.write(f"track_id:{track_id}\n")
            f.write(f"pose_status:{pose_status}\n")
        
        if pose_status != "NORMAL":
            print(f"[POSE] ID:{track_id} -> {pose_status}")
        
        os.remove(job_path)
    except Exception as e:
        print(f"[POSE] Job error: {e}")
        try:
            os.remove(job_path)
        except: pass
    
    time.sleep(0.05)