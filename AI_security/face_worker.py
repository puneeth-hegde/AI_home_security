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

# --- TUNED SETTINGS ---
# 0.22 is the sweet spot for angled cameras + Voting
STRICT_THRESHOLD = 0.22 
# Voting: Needs 3 matches in the last 5 frames to confirm identity
VOTE_HISTORY = 5
VOTE_REQUIRED = 3

# Create Dirs
for d in [JOBS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE] Loading Model...")
DeepFace.build_model(MODEL_NAME)
print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE] VOTING ENGINE READY.")

# Memory: { 'door_1': deque(['puneeth', 'Unknown', ...]) }
vote_memory = {}

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
                # 1. RECOGNIZE (Single Frame)
                # enforce_detection=False is CRITICAL for your high camera angle
                dfs = DeepFace.find(img_path=job_path, db_path=DB_PATH, model_name=MODEL_NAME, 
                                  detector_backend=DETECTOR_BACKEND, distance_metric="cosine", 
                                  enforce_detection=False, silent=True)
                
                raw_id = "Unknown"
                dist = 1.0
                
                if len(dfs) > 0 and not dfs[0].empty:
                    dist = dfs[0].iloc[0]["distance"]
                    
                    if dist <= STRICT_THRESHOLD:
                        path = dfs[0].iloc[0]["identity"]
                        # Handle folder structure (dataset/puneeth/img.jpg)
                        if os.path.sep in path: 
                            raw_id = os.path.basename(os.path.dirname(path))
                        else: 
                            raw_id = os.path.basename(path).split('.')[0]

                # 2. VOTING LOGIC (Stability)
                if job_id not in vote_memory: 
                    vote_memory[job_id] = deque(maxlen=VOTE_HISTORY)
                
                vote_memory[job_id].append(raw_id)
                
                # Count votes
                counts = Counter(vote_memory[job_id])
                most_common, count = counts.most_common(1)[0]
                
                # Decision
                if count >= VOTE_REQUIRED:
                    final_id = most_common
                else:
                    final_id = "Verifying..."

                # VERBOSE LOG
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE] {job_id} Raw: {raw_id} ({dist:.3f}) -> Voted: {final_id} ({count}/{len(vote_memory[job_id])})")

                with open(result_path, "w") as f: f.write(final_id)

            except Exception as e:
                # print(f"Error: {e}")
                pass
            
            # Cleanup
            try: os.remove(job_path)
            except: pass

    except KeyboardInterrupt: break
    except Exception: time.sleep(0.1)