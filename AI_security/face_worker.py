# face_worker.py - OPTIMIZED FACE RECOGNITION
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import glob
import cv2
import numpy as np
import warnings
from deepface import DeepFace
from collections import deque, Counter
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# CONFIGURATION
JOBS_DIR = "jobs_face"
RESULTS_DIR = "results_face"
DB_PATH = "dataset"
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"

# Door camera = 70 degrees, needs relaxed threshold
CAMERA_PROFILES = {
    'door': {'threshold': 0.35, 'vote_required': 4, 'vote_history': 6},
    'default': {'threshold': 0.22, 'vote_required': 3, 'vote_history': 5}
}

BLUR_VAR_THRESHOLD = 80.0
BRIGHTNESS_THRESHOLD = 45.0
MIN_FACE_SIZE = 60

# LOGGING
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger('face_worker')
handler = logging.FileHandler('logs/face_worker.log')
handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('[%(asctime)s] [FACE] %(message)s'))
logger.addHandler(console)

for d in [JOBS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

logger.info("Loading DeepFace Model...")
DeepFace.build_model(MODEL_NAME)
logger.info("FACE RECOGNITION ENGINE READY")

vote_memory = {}

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_image_quality(img, job_id):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return False, "ERROR:TOO_SMALL", {}
    
    blur_var = variance_of_laplacian(gray)
    if blur_var < BLUR_VAR_THRESHOLD:
        logger.info(f"{job_id} REJECTED: BLUR blur_var={blur_var:.1f}")
        return False, "ERROR:BLURRY", {"blur_var": round(blur_var, 2)}
    
    mean_brightness = float(np.mean(gray))
    if mean_brightness < BRIGHTNESS_THRESHOLD:
        logger.info(f"{job_id} REJECTED: DARK brightness={mean_brightness:.1f}")
        return False, "ERROR:DARK", {"brightness": round(mean_brightness, 1)}
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    if edge_density < 0.05 or edge_density > 0.25:
        logger.warning(f"{job_id} SUSPECTED_SPOOF: edge_density={edge_density:.3f}")
        return False, "ERROR:SPOOF:TEXTURE", {"edge_density": round(edge_density, 3)}
    
    return True, "OK", {"blur_var": round(blur_var, 2), "brightness": round(mean_brightness, 1)}

def recognize_face(img, job_path, job_id):
    if 'door' in job_id.lower():
        profile = CAMERA_PROFILES['door']
    else:
        profile = CAMERA_PROFILES['default']
    
    try:
        dfs = DeepFace.find(
            img_path=job_path, db_path=DB_PATH, model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND, distance_metric="cosine",
            enforce_detection=False, silent=True
        )
        
        raw_id = "Unknown"
        dist = 1.0
        
        if len(dfs) > 0 and not dfs[0].empty:
            dist = float(dfs[0].iloc[0]["distance"])
            if dist <= profile['threshold']:
                path = dfs[0].iloc[0]["identity"]
                if os.path.sep in path:
                    raw_id = os.path.basename(os.path.dirname(path))
                else:
                    raw_id = os.path.basename(path).split('.')[0]
        
        if job_id not in vote_memory:
            vote_memory[job_id] = deque(maxlen=profile['vote_history'])
        
        vote_memory[job_id].append(raw_id)
        counts = Counter(vote_memory[job_id])
        most_common, count = counts.most_common(1)[0]
        
        if count >= profile['vote_required']:
            final_id = most_common
        else:
            final_id = "Verifying..."
        
        logger.info(f"{job_id} RAW={raw_id} dist={dist:.3f} VOTED={final_id} votes={count}/{len(vote_memory[job_id])}")
        return final_id
    
    except Exception as e:
        logger.error(f"{job_id} RECOGNITION_ERROR: {e}")
        return "Error"

logger.info("Entering Main Loop...")
processed_count = 0

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
                img = cv2.imread(job_path)
                if img is None:
                    logger.error(f"{job_id} INVALID_IMAGE")
                    with open(result_path, "w") as f:
                        f.write("Error")
                    continue
                
                is_valid, error_code, quality_meta = check_image_quality(img, job_id)
                
                if not is_valid:
                    with open(result_path, "w") as f:
                        f.write(error_code)
                    try:
                        os.remove(job_path)
                    except:
                        pass
                    continue
                
                identity = recognize_face(img, job_path, job_id)
                
                with open(result_path, "w") as f:
                    f.write(identity)
                
                processed_count += 1
                
                if processed_count % 50 == 0:
                    logger.info(f"PERFORMANCE: {processed_count} faces processed")
            
            except Exception as e:
                logger.error(f"{job_id} EXCEPTION: {e}")
                with open(result_path, "w") as f:
                    f.write("Error")
            
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

logger.info(f"SHUTDOWN: {processed_count} total faces processed")