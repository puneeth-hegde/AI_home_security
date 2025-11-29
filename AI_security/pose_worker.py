# pose_worker.py - OPTIMIZED POSE ANALYSIS
import os
import time
import cv2
import glob
import math
import mediapipe as mp
from collections import deque
from datetime import datetime
import logging

# CONFIGURATION
JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"

VIOLENCE_SPEED_THRESHOLD = 0.025
ACCELERATION_THRESHOLD = 0.015
HISTORY_SIZE = 10

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# LOGGING
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger('pose_worker')
handler = logging.FileHandler('logs/pose_worker.log')
handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('[%(asctime)s] [POSE] %(message)s'))
logger.addHandler(console)

for d in [JOBS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

logger.info("POSE ANALYSIS ENGINE READY")

tracking_memory = {}
processed_count = 0

def extract_keypoints(landmarks):
    try:
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        return {
            'nose': (nose.x, nose.y, nose.visibility),
            'left_wrist': (left_wrist.x, left_wrist.y, left_wrist.visibility),
            'right_wrist': (right_wrist.x, right_wrist.y, right_wrist.visibility),
            'left_hip': (left_hip.x, left_hip.y, left_hip.visibility),
            'right_hip': (right_hip.x, right_hip.y, right_hip.visibility),
            'hips_y': (left_hip.y + right_hip.y) / 2.0
        }
    except:
        return None

def analyze_violence(keypoints, job_id):
    if not keypoints:
        return "Normal", 0.0, 0.0, {}
    
    nose_y = keypoints['nose'][1]
    left_wrist_y = keypoints['left_wrist'][1]
    right_wrist_y = keypoints['right_wrist'][1]
    
    if (left_wrist_y < nose_y and keypoints['left_wrist'][2] > 0.5 and
        right_wrist_y < nose_y and keypoints['right_wrist'][2] > 0.5):
        return "Threat: Surrender", 0.0, 0.0, {"pose": "hands_up"}
    
    current_data = {
        'left_wrist': (keypoints['left_wrist'][0], keypoints['left_wrist'][1]),
        'right_wrist': (keypoints['right_wrist'][0], keypoints['right_wrist'][1]),
        'hips_y': keypoints['hips_y'],
        'timestamp': time.time()
    }
    
    if job_id not in tracking_memory:
        tracking_memory[job_id] = deque(maxlen=HISTORY_SIZE)
    
    tracking_memory[job_id].append(current_data)
    
    if len(tracking_memory[job_id]) < 3:
        return "Normal", 0.0, 0.0, {"history": len(tracking_memory[job_id])}
    
    history = list(tracking_memory[job_id])
    start = history[-3]
    end = history[-1]
    dt = end['timestamp'] - start['timestamp']
    
    if dt < 0.001 or dt > 2.0:
        return "Normal", 0.0, 0.0, {"dt": "invalid"}
    
    left_dist = math.hypot(start['left_wrist'][0] - end['left_wrist'][0], start['left_wrist'][1] - end['left_wrist'][1])
    right_dist = math.hypot(start['right_wrist'][0] - end['right_wrist'][0], start['right_wrist'][1] - end['right_wrist'][1])
    max_speed = max(left_dist, right_dist) / dt
    
    acceleration = 0.0
    if len(history) >= 5:
        start1 = history[-5]
        end1 = history[-3]
        dt1 = end1['timestamp'] - start1['timestamp']
        start2 = history[-3]
        end2 = history[-1]
        dt2 = end2['timestamp'] - start2['timestamp']
        
        if dt1 > 0.001 and dt2 > 0.001:
            dist1 = max(math.hypot(start1['left_wrist'][0] - end1['left_wrist'][0], start1['left_wrist'][1] - end1['left_wrist'][1]),
                       math.hypot(start1['right_wrist'][0] - end1['right_wrist'][0], start1['right_wrist'][1] - end1['right_wrist'][1]))
            speed1 = dist1 / dt1
            dist2 = max(math.hypot(start2['left_wrist'][0] - end2['left_wrist'][0], start2['left_wrist'][1] - end2['left_wrist'][1]),
                       math.hypot(start2['right_wrist'][0] - end2['right_wrist'][0], start2['right_wrist'][1] - end2['right_wrist'][1]))
            speed2 = dist2 / dt2
            acceleration = abs(speed2 - speed1) / ((dt1 + dt2) / 2)
    
    status = "Normal"
    if acceleration > ACCELERATION_THRESHOLD:
        status = "Threat: Violent Motion"
        logger.warning(f"{job_id} VIOLENCE_DETECTED accel={acceleration:.4f} speed={max_speed:.4f}")
    elif max_speed > VIOLENCE_SPEED_THRESHOLD:
        status = "Threat: Fast Movement"
        logger.warning(f"{job_id} FAST_MOVEMENT speed={max_speed:.4f}")
    
    crawler_hint = current_data['hips_y'] > 0.78
    metadata = {"hips_y": round(current_data['hips_y'], 3), "history_len": len(tracking_memory[job_id]), "crawler_hint": crawler_hint}
    
    if status == "Normal" and crawler_hint:
        status = "Crawler"
    
    return status, max_speed, acceleration, metadata

logger.info("Entering Main Loop...")

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
                if image is None:
                    raise Exception("Invalid image")
                
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                status = "Normal"
                speed = 0.0
                acceleration = 0.0
                metadata = {}
                
                if results.pose_landmarks:
                    keypoints = extract_keypoints(results.pose_landmarks.landmark)
                    if keypoints:
                        status, speed, acceleration, metadata = analyze_violence(keypoints, job_id)
                        if status != "Normal":
                            logger.info(f"{job_id} STATUS={status} speed={speed:.4f} accel={acceleration:.4f}")
                else:
                    metadata = {"error": "no_pose_detected"}
                
                meta_str = ",".join([f"{k}={v}" for k, v in metadata.items()])
                output = f"{status}|{speed:.4f}|{meta_str}"
                
                with open(result_path, "w") as f:
                    f.write(output)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"PERFORMANCE: {processed_count} poses analyzed")
            
            except Exception as e:
                logger.error(f"{job_id} EXCEPTION: {e}")
                with open(result_path, "w") as f:
                    f.write("Error|0.0|error=exception")
            
            try:
                os.remove(job_path)
            except:
                pass
    
    except KeyboardInterrupt:
        logger.info("SHUTDOWN_REQUESTED")
        break
    except Exception as e:
        logger.error(f"MAIN_LOOP_EXCEPTION: {e}")
        time.sleep(0.1)

logger.info(f"SHUTDOWN: {processed_count} total poses analyzed")