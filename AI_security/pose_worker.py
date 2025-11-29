# pose_worker_v2.py - ENHANCED POSE DETECTION WITH WRIST TRACKING
import os
import time
import cv2
import glob
import math
import mediapipe as mp
from collections import deque
import logging
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"

# --- ACTION RECOGNITION THRESHOLDS ---
VIOLENCE_VELOCITY_THRESH = 1.5   
WAVE_VELOCITY_THRESH = 0.5       
HISTORY_SIZE = 10                

# ============================================================================
# SETUP & LOGGING
# ============================================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, 
    model_complexity=1, 
    enable_segmentation=False, 
    min_detection_confidence=0.5
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/pose_worker.log', 
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger()

for d in [JOBS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

logger.info("=" * 80)
logger.info("POSE WORKER v2.0 - WITH WRIST TRACKING")
logger.info("=" * 80)

tracking_memory = {}

# ============================================================================
# MATH HELPERS
# ============================================================================
def calculate_distance(point1, point2):
    """Calculates Euclidean distance between two (x,y) points."""
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return math.hypot(dx, dy)

# ============================================================================
# ANALYSIS LOGIC
# ============================================================================
def analyze_pose(landmarks, job_id, img_height):
    """
    Analyzes body landmarks for specific behaviors.
    Returns: (status, velocity, metadata_dict)
    """
    # Extract relevant landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    # Calculate average positions
    avg_hips_y = (left_hip.y + right_hip.y) / 2.0
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    
    # ====================================================================
    # 1. SURRENDER DETECTION (Hands Up)
    # ====================================================================
    # Both wrists above nose AND above shoulders
    hands_above_nose = (left_wrist.y < nose.y) and (right_wrist.y < nose.y)
    hands_above_shoulders = (left_wrist.y < avg_shoulder_y) and (right_wrist.y < avg_shoulder_y)
    
    # Check if arms are extended (not just touching head)
    left_arm_extended = abs(left_wrist.x - left_shoulder.x) > 0.1
    right_arm_extended = abs(right_wrist.x - right_shoulder.x) > 0.1
    
    if hands_above_nose and hands_above_shoulders and (left_arm_extended or right_arm_extended):
        metadata = {
            "hips_y": avg_hips_y,
            "left_wrist_x": left_wrist.x,
            "left_wrist_y": left_wrist.y,
            "right_wrist_x": right_wrist.x,
            "right_wrist_y": right_wrist.y,
            "hands_up": True
        }
        return "SURRENDER", 0.0, metadata

    # ====================================================================
    # 2. VELOCITY / VIOLENCE ANALYSIS
    # ====================================================================
    current_time = time.time()
    
    if job_id not in tracking_memory:
        tracking_memory[job_id] = deque(maxlen=HISTORY_SIZE)
    
    tracking_memory[job_id].append({
        't': current_time,
        'lw': (left_wrist.x, left_wrist.y),
        'rw': (right_wrist.x, right_wrist.y)
    })

    status = "Normal"
    max_velocity = 0.0
    
    if len(tracking_memory[job_id]) >= 3:
        prev_state = tracking_memory[job_id][-3]
        curr_state = tracking_memory[job_id][-1]
        
        time_diff = curr_state['t'] - prev_state['t']
        
        if time_diff > 0.01:
            left_dist = calculate_distance(prev_state['lw'], curr_state['lw'])
            right_dist = calculate_distance(prev_state['rw'], curr_state['rw'])
            
            left_velocity = left_dist / time_diff
            right_velocity = right_dist / time_diff
            
            max_velocity = max(left_velocity, right_velocity)
            
            if max_velocity > VIOLENCE_VELOCITY_THRESH:
                status = "VIOLENCE"
                logger.warning(f"[{job_id}] VIOLENCE DETECTED! Velocity: {max_velocity:.2f}")
            elif max_velocity > WAVE_VELOCITY_THRESH:
                status = "WAVING"

    # ====================================================================
    # 3. BUILD METADATA (CRITICAL: Include wrist positions for weapon check)
    # ====================================================================
    metadata = {
        "hips_y": avg_hips_y,
        "left_wrist_x": left_wrist.x,
        "left_wrist_y": left_wrist.y,
        "right_wrist_x": right_wrist.x,
        "right_wrist_y": right_wrist.y,
        "left_elbow_x": left_elbow.x,
        "left_elbow_y": left_elbow.y,
        "right_elbow_x": right_elbow.x,
        "right_elbow_y": right_elbow.y,
        "velocity": max_velocity
    }

    return status, max_velocity, metadata

# ============================================================================
# MAIN LOOP
# ============================================================================
logger.info("Entering main loop...")
processed_count = 0

while True:
    try:
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        
        if not job_files:
            time.sleep(0.01)
            continue

        for job_path in job_files:
            job_filename = os.path.basename(job_path)
            job_id = job_filename.split('.')[0]
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            
            try:
                img = cv2.imread(job_path)
                if img is None:
                    raise Exception("Failed to read image")
                
                img_height = img.shape[0]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = pose.process(img_rgb)
                
                status = "Normal"
                velocity = 0.0
                metadata = {}
                
                if results.pose_landmarks:
                    status, velocity, metadata = analyze_pose(
                        results.pose_landmarks.landmark, 
                        job_id, 
                        img_height
                    )
                else:
                    status = "NoPose"
                    logger.info(f"[{job_id}] No pose detected")

                # Format output as JSON-like string (easier parsing)
                # Format: STATUS|VELOCITY|key1=val1,key2=val2,...
                meta_parts = []
                for key, val in metadata.items():
                    if isinstance(val, float):
                        meta_parts.append(f"{key}={val:.4f}")
                    elif isinstance(val, bool):
                        meta_parts.append(f"{key}={int(val)}")
                
                meta_str = ",".join(meta_parts)
                output_str = f"{status}|{velocity:.2f}|{meta_str}"
                
                with open(result_path, "w") as f:
                    f.write(output_str)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"MILESTONE: {processed_count} poses processed")

            except Exception as e:
                logger.error(f"Error processing {job_id}: {e}")
                with open(result_path, "w") as f:
                    f.write("Error|0.0|error=1")
            
            try:
                os.remove(job_path)
            except:
                pass

    except KeyboardInterrupt:
        logger.info("SHUTDOWN_REQUESTED")
        break
    except Exception as e:
        logger.error(f"Loop Error: {e}")
        time.sleep(1)

logger.info(f"SHUTDOWN: {processed_count} total poses processed")