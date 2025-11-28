import cv2
import mediapipe as mp
import os
import glob
import time
from datetime import datetime
import warnings
import shutil

# --- Suppress Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning) 

# --- Configuration ---
JOBS_DIR = "jobs_pose"
RESULTS_DIR = "results_pose"
PROCESSING_DIR = "jobs_pose_processing" # --- FIX: New processing directory
POLLING_INTERVAL = 0.1

# --- Load MediaPipe Model ONCE at Startup ---
print(f"[{datetime.now()}] [POSE_WORKER] Starting...")
print(f"[{datetime.now()}] [POSE_WORKER] Loading MediaPipe Pose model into memory...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
print(f"[{datetime.now()}] [POSE_WORKER] Model loaded.")

# --- Ensure job/result directories exist ---
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True) # --- FIX: Create processing dir

# --- Main Worker Loop ---
print(f"[{datetime.now()}] [POSE_WORKER] Worker is now running. Watching for jobs in '{JOBS_DIR}'...")
while True:
    try:
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        
        if not job_files:
            time.sleep(POLLING_INTERVAL)
            continue

        for job_path in job_files:
            job_filename = os.path.basename(job_path)
            job_id = job_filename.split('.')[0]
            result_filename = f"result_{job_id}.txt"
            result_path = os.path.join(RESULTS_DIR, result_filename)
            
            # --- FIX: Atomic File Operation ---
            processing_path = os.path.join(PROCESSING_DIR, job_filename)
            try:
                shutil.move(job_path, processing_path)
            except Exception as e:
                continue
            # --- END FIX ---
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE_WORKER] New job received: {job_id}")
            pose_status = "Normal" 
            
            try:
                frame = cv2.imread(processing_path)
                if frame is None:
                    raise ValueError("Could not read image file")
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    
                    if (left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5 and
                        left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5): 
                        if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):
                            pose_status = "Threat: Hands Raised"
                
                with open(result_path, "w") as f:
                    f.write(pose_status)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE_WORKER] Job complete. Result: {pose_status}")

            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE_WORKER] Error processing {job_id}: {e}")
                with open(result_path, "w") as f:
                    f.write("Error") 
            
            os.remove(processing_path) # Clean up

    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE_WORKER] Shutdown signal received. Exiting.")
        break
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [POSE_WORKER] An unexpected error occurred: {e}")
        time.sleep(1)