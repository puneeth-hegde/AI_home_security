import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import glob
import shutil
import warnings
from deepface import DeepFace
from datetime import datetime
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# --- CONFIGURATION ---
JOBS_DIR = "jobs_face"
RESULTS_DIR = "results_face"
PROCESSING_DIR = "jobs_face_processing"
DB_PATH = "dataset" 
MODEL_NAME = "Facenet512"

# --- FAIL-SAFE SETTINGS ---
# We use OpenCV because it is the only one that handles steep angles well.
DETECTOR_BACKEND = "opencv" 

# 0.20 is the balance point. 
# It rejects friends (usually 0.25+) but accepts you (usually 0.10-0.15).
STRICT_THRESHOLD = 0.20

POLLING_INTERVAL = 0.05 

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Starting...")
# Force build to ensure weights are ready
DeepFace.build_model(MODEL_NAME)
print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Ready.")

while True:
    try:
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        if not job_files:
            time.sleep(POLLING_INTERVAL)
            continue

        for job_path in job_files:
            job_filename = os.path.basename(job_path)
            job_id = job_filename.split('.')[0] 
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            processing_path = os.path.join(PROCESSING_DIR, job_filename)
            
            try:
                shutil.move(job_path, processing_path)
            except: continue 

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Processing: {job_id}")
            identity = "Unknown"
            
            try:
                # --- FAIL-SAFE RECOGNITION ---
                # enforce_detection=False allows the AI to "Try its best" even on angled faces
                dfs = DeepFace.find(
                    img_path=processing_path,
                    db_path=DB_PATH,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    distance_metric="cosine", 
                    enforce_detection=False, 
                    silent=True
                )
                
                if len(dfs) > 0 and not dfs[0].empty:
                    distance = dfs[0].iloc[0]["distance"]
                    
                    if distance <= STRICT_THRESHOLD:
                        full_path = dfs[0].iloc[0]["identity"]
                        # Robust naming
                        parent = os.path.dirname(full_path)
                        if os.path.abspath(parent) == os.path.abspath(DB_PATH):
                            identity = os.path.basename(full_path).split('.')[0]
                        else:
                            identity = os.path.basename(parent)
                        print(f"   >>> MATCH: {identity} (Dist: {distance:.4f})")
                    else:
                        print(f"   >>> REJECTED: Distance {distance:.4f} > {STRICT_THRESHOLD}")

                with open(result_path, "w") as f: f.write(identity)

            except Exception as e:
                # print(f"Error: {e}") 
                with open(result_path, "w") as f: f.write("Error")
            
            if os.path.exists(processing_path): os.remove(processing_path)

    except KeyboardInterrupt: break
    except Exception: time.sleep(0.1)