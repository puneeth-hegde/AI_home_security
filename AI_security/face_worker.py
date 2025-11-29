import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import glob
import cv2
import numpy as np
from deepface import DeepFace
from collections import deque, Counter

JOBS_DIR = "jobs_face"
RESULTS_DIR = "results_face"
DB_PATH = "dataset"
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
STRICT_THRESHOLD = 0.30  # Relaxed for steep door camera angles (was 0.22)

BLUR_THRESHOLD = 100
BRIGHTNESS_LOW = 50

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[FACE] Building model...")
try:
    DeepFace.build_model(MODEL_NAME)
    print("[FACE] Ready")
except Exception as e:
    print(f"[FACE] Error: {e}")

vote_memory = {}

def check_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < BLUR_THRESHOLD

def check_darkness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < BRIGHTNESS_LOW

def assess_quality(img):
    if check_blur(img):
        return "BLUR"
    elif check_darkness(img):
        return "DARK"
    else:
        return "OK"

def recognize_with_voting(track_id, img):
    quality = assess_quality(img)
    if quality != "OK":
        return "Unknown", quality
    
    if track_id not in vote_memory:
        vote_memory[track_id] = deque(maxlen=5)
    
    try:
        dfs = DeepFace.find(
            img_path=img,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric="cosine",
            enforce_detection=False,
            silent=True
        )
        
        identity = "Unknown"
        if len(dfs) > 0 and len(dfs[0]) > 0:
            best_match = dfs[0].iloc[0]
            distance = best_match['distance']
            if distance < STRICT_THRESHOLD:
                identity_path = best_match['identity']
                identity = identity_path.split(os.sep)[-2]
        
        vote_memory[track_id].append(identity)
        
        if len(vote_memory[track_id]) >= 3:
            vote_counts = Counter(vote_memory[track_id])
            top_identity, top_count = vote_counts.most_common(1)[0]
            if top_count >= 3:
                return top_identity, quality
        
        return identity, quality
    except Exception as e:
        print(f"[FACE] Recognition error: {e}")
        return "Unknown", quality

print("[FACE] Worker started")

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
        
        identity, quality = recognize_with_voting(track_id, img)
        
        result_filename = f"{track_id}_{int(time.time()*1000)}.txt"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        with open(result_path, 'w') as f:
            f.write(f"track_id:{track_id}\n")
            f.write(f"identity:{identity}\n")
            f.write(f"quality:{quality}\n")
        
        print(f"[FACE] ID:{track_id} -> {identity} (Quality: {quality})")
        os.remove(job_path)
    except Exception as e:
        print(f"[FACE] Job error: {e}")
        try:
            os.remove(job_path)
        except: pass
    
    time.sleep(0.05)