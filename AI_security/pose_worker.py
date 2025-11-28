import os
import time
import cv2
import glob
import math
import numpy as np
import mediapipe as mp
from collections import deque

# Configuration
JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"

# Detection thresholds
SURRENDER_THRESHOLD = 0.15  # Wrist height relative to head
VIOLENCE_SPEED_THRESHOLD = 0.06  # Movement speed for punch detection
VIOLENCE_ACCELERATION_THRESHOLD = 0.03  # Sudden acceleration

# Temporal smoothing
HISTORY_SIZE = 5  # Frames to keep in memory
MIN_CONFIDENCE = 0.5  # Minimum landmark confidence

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,  # Higher complexity for better accuracy
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Tracking memory: {job_id: deque of pose data}
tracking_memory = {}

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_angle(a, b, c):
    """Calculate angle between three points (b is the vertex)"""
    ba = [a.x - b.x, a.y - b.y]
    bc = [c.x - b.x, c.y - b.y]
    
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    if mag_ba * mag_bc == 0:
        return 0
    
    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.acos(cos_angle)
    
    return math.degrees(angle)

def check_surrender_pose(landmarks):
    """
    Detect surrender pose: Both hands raised above head
    Returns: (is_surrender, confidence)
    """
    # Key landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Check visibility
    if (left_wrist.visibility < MIN_CONFIDENCE or 
        right_wrist.visibility < MIN_CONFIDENCE):
        return False, 0.0
    
    # Calculate relative positions
    head_y = nose.y
    left_wrist_y = left_wrist.y
    right_wrist_y = right_wrist.y
    
    # Both wrists should be above head
    left_raised = (head_y - left_wrist_y) > SURRENDER_THRESHOLD
    right_raised = (head_y - right_wrist_y) > SURRENDER_THRESHOLD
    
    if not (left_raised and right_raised):
        return False, 0.0
    
    # Check elbow angles (should be somewhat extended, not fully bent)
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Arms should be raised (angles between 120-180 degrees)
    left_extended = 120 < left_angle < 180
    right_extended = 120 < right_angle < 180
    
    # Calculate confidence
    confidence = 0.0
    if left_raised:
        confidence += 0.3
    if right_raised:
        confidence += 0.3
    if left_extended:
        confidence += 0.2
    if right_extended:
        confidence += 0.2
    
    is_surrender = left_raised and right_raised and (left_extended or right_extended)
    
    return is_surrender, confidence

def check_violent_motion(job_id, landmarks):
    """
    Detect violent motion: Fast punching movements
    Uses temporal data to detect rapid hand movements
    Returns: (is_violent, confidence, description)
    """
    # Key landmarks
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    
    # Check visibility
    if (left_wrist.visibility < MIN_CONFIDENCE and 
        right_wrist.visibility < MIN_CONFIDENCE):
        return False, 0.0, ""
    
    # Current positions
    current_data = {
        'left_wrist': (left_wrist.x, left_wrist.y),
        'right_wrist': (right_wrist.x, right_wrist.y),
        'left_elbow': (left_elbow.x, left_elbow.y),
        'right_elbow': (right_elbow.x, right_elbow.y),
        'timestamp': time.time()
    }
    
    # Initialize or update tracking memory
    if job_id not in tracking_memory:
        tracking_memory[job_id] = deque(maxlen=HISTORY_SIZE)
    
    tracking_memory[job_id].append(current_data)
    
    # Need at least 3 frames for motion analysis
    if len(tracking_memory[job_id]) < 3:
        return False, 0.0, ""
    
    # Analyze motion
    history = list(tracking_memory[job_id])
    
    # Calculate velocities and accelerations
    left_speeds = []
    right_speeds = []
    
    for i in range(1, len(history)):
        prev = history[i-1]
        curr = history[i]
        dt = curr['timestamp'] - prev['timestamp']
        
        if dt > 0:
            # Left wrist speed
            dx = curr['left_wrist'][0] - prev['left_wrist'][0]
            dy = curr['left_wrist'][1] - prev['left_wrist'][1]
            left_speed = math.sqrt(dx*dx + dy*dy) / dt
            left_speeds.append(left_speed)
            
            # Right wrist speed
            dx = curr['right_wrist'][0] - prev['right_wrist'][0]
            dy = curr['right_wrist'][1] - prev['right_wrist'][1]
            right_speed = math.sqrt(dx*dx + dy*dy) / dt
            right_speeds.append(right_speed)
    
    # Check for high-speed movements
    max_left_speed = max(left_speeds) if left_speeds else 0
    max_right_speed = max(right_speeds) if right_speeds else 0
    
    # Calculate acceleration (change in speed)
    left_accel = 0
    right_accel = 0
    if len(left_speeds) >= 2:
        left_accel = abs(left_speeds[-1] - left_speeds[-2])
        right_accel = abs(right_speeds[-1] - right_speeds[-2])
    
    # Detect violence
    is_violent = False
    confidence = 0.0
    description = ""
    
    # High speed punch
    if max_left_speed > VIOLENCE_SPEED_THRESHOLD or max_right_speed > VIOLENCE_SPEED_THRESHOLD:
        is_violent = True
        confidence += 0.4
        description = "FAST_MOVEMENT"
    
    # Sudden acceleration (punch-like)
    if left_accel > VIOLENCE_ACCELERATION_THRESHOLD or right_accel > VIOLENCE_ACCELERATION_THRESHOLD:
        is_violent = True
        confidence += 0.4
        description = "PUNCH_MOTION"
    
    # Check arm extension (punching pose)
    latest = history[-1]
    left_ext = math.sqrt(
        (latest['left_wrist'][0] - latest['left_elbow'][0])**2 +
        (latest['left_wrist'][1] - latest['left_elbow'][1])**2
    )
    right_ext = math.sqrt(
        (latest['right_wrist'][0] - latest['right_elbow'][0])**2 +
        (latest['right_wrist'][1] - latest['right_elbow'][1])**2
    )
    
    # Extended arm with high speed indicates punch
    if (left_ext > 0.15 or right_ext > 0.15) and is_violent:
        confidence += 0.2
        description = "AGGRESSIVE_PUNCH"
    
    return is_violent, min(confidence, 1.0), description

def check_weapon_holding_pose(landmarks):
    """
    Detect potential weapon-holding poses
    Returns: (is_holding, confidence)
    """
    # Key landmarks
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Check if both hands are close together (two-handed weapon grip)
    wrist_distance = calculate_distance(left_wrist, right_wrist)
    
    # Check if hands are in front of body (holding pose)
    avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    avg_wrist_x = (left_wrist.x + right_wrist.x) / 2
    
    # Hands close together and in front
    is_holding = (wrist_distance < 0.15 and 
                  abs(avg_wrist_x - avg_shoulder_x) < 0.2)
    
    confidence = 0.0
    if wrist_distance < 0.15:
        confidence += 0.5
    if abs(avg_wrist_x - avg_shoulder_x) < 0.2:
        confidence += 0.5
    
    return is_holding, confidence

def process_job(job_path, job_id):
    """Process a single pose detection job"""
    try:
        image = cv2.imread(job_path)
        if image is None:
            return "Error"
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return "NO_POSE_DETECTED"
        
        landmarks = results.pose_landmarks.landmark
        
        # Check surrender pose
        is_surrender, surrender_conf = check_surrender_pose(landmarks)
        
        # Check violent motion
        is_violent, violence_conf, violence_desc = check_violent_motion(job_id, landmarks)
        
        # Check weapon holding pose
        is_weapon_hold, weapon_conf = check_weapon_holding_pose(landmarks)
        
        # Determine final status with priority
        if is_surrender and surrender_conf > 0.6:
            return f"SURRENDER (Conf: {surrender_conf:.2f})"
        
        if is_violent and violence_conf > 0.5:
            return f"VIOLENCE-{violence_desc} (Conf: {violence_conf:.2f})"
        
        if is_weapon_hold and weapon_conf > 0.6:
            return f"WEAPON_POSE (Conf: {weapon_conf:.2f})"
        
        # Check for unusual poses
        if is_surrender and surrender_conf > 0.3:
            return f"Hands_Raised (Conf: {surrender_conf:.2f})"
        
        if is_violent and violence_conf > 0.3:
            return f"Fast_Motion (Conf: {violence_conf:.2f})"
        
        return "Normal"
    
    except Exception as e:
        print(f"Error processing pose for {job_id}: {e}")
        return "Error"

def cleanup_old_tracking():
    """Remove tracking data for jobs that are no longer active"""
    current_time = time.time()
    to_remove = []
    
    for job_id, history in tracking_memory.items():
        if len(history) > 0:
            latest_time = history[-1]['timestamp']
            if current_time - latest_time > 10:  # 10 seconds timeout
                to_remove.append(job_id)
    
    for job_id in to_remove:
        del tracking_memory[job_id]

# Main worker loop
print("Pose detection worker started")
print(f"Violence speed threshold: {VIOLENCE_SPEED_THRESHOLD}")
print(f"Surrender threshold: {SURRENDER_THRESHOLD}")

last_cleanup = time.time()

while True:
    job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
    
    if not job_files:
        time.sleep(0.05)
        
        # Periodic cleanup
        if time.time() - last_cleanup > 30:
            cleanup_old_tracking()
            last_cleanup = time.time()
        
        continue
    
    for job_path in job_files:
        job_id = os.path.basename(job_path).split('.')[0]
        
        # Process the job
        status = process_job(job_path, job_id)
        
        # Write result
        try:
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            with open(result_path, "w") as f:
                f.write(status)
            
            # Log for debugging
            if status not in ["Normal", "Error", "NO_POSE_DETECTED"]:
                print(f"{job_id}: {status}")
        except Exception as e:
            print(f"Error writing result: {e}")
        
        # Clean up job file
        try:
            os.remove(job_path)
        except:
            pass