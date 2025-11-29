import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import glob
import cv2
import numpy as np
import warnings
from deepface import DeepFace
from collections import deque, Counter
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
JOBS_DIR = "jobs_face"
RESULTS_DIR = "results_face"
DB_PATH = "dataset" 
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"

# --- VOTING CONFIGURATION ---
# We need 3 out of 5 frames to agree to confirm an identity.
VOTE_HISTORY_LENGTH = 5
VOTES_REQUIRED = 3

# Strictness (0.22 allows for your steep angle)
STRICT_THRESHOLD = 0.22

# Memory for Voting: { 'door_1': ['puneeth', 'Unknown', 'puneeth'...] }
vote_memory = {}

DeepFace.build_model(MODEL_NAME)
print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Voting Engine Ready.")

def get_voted_result(job_id, current_result):
    # Initialize memory for this person if new
    if job_id not in vote_memory:
        vote_memory[job_id] = deque(maxlen=VOTE_HISTORY_LENGTH)
    
    # Add new result to history
    vote_memory[job_id].append(current_result)
    
    # Count votes
    votes = list(vote_memory[job_id])
    counts = Counter(votes)
    most_common, count = counts.most_common(1)[0]
    
    # Debug: Print the voting process so you can see it working
    print(f"   >>> Voting Logic [{job_id}]: {votes} -> Winner: {most_common} ({count}/{len(votes)})")
    
    # Decision
    if count >= VOTES_REQUIRED:
        return most_common
    else:
        return "Verifying..." # Not enough confidence yet

while True:
    try:
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        if not job_files:
            time.sleep(0.05)
            continue

        for job_path in job_files:
            job_id = os.path.basename(job_path).split('.')[0]
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            
            try:
                # 1. Recognize Single Frame
                dfs = DeepFace.find(img_path=job_path, db_path=DB_PATH, model_name=MODEL_NAME, 
                                  detector_backend=DETECTOR_BACKEND, distance_metric="cosine", 
                                  enforce_detection=False, silent=True)
                
                raw_identity = "Unknown"
                if len(dfs) > 0 and not dfs[0].empty:
                    dist = dfs[0].iloc[0]["distance"]
                    if dist <= STRICT_THRESHOLD:
                        full_path = dfs[0].iloc[0]["identity"]
                        # Extract Name
                        if os.path.sep in full_path:
                            raw_identity = os.path.basename(os.path.dirname(full_path))
                        else:
                            raw_identity = os.path.basename(full_path).split('.')[0]

                # 2. Apply Voting Logic
                final_identity = get_voted_result(job_id, raw_identity)
                
                # Only write to file if we have a stable result
                with open(result_path, "w") as f: f.write(final_identity)

            except Exception as e:
                # print(f"Error: {e}")
                pass
            
            # Clean up
            try: os.remove(job_path)
            except: pass

    except KeyboardInterrupt: break
    except Exception: time.sleep(0.1)