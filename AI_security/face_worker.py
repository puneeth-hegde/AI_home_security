import os
# --- CONFIGURATION: FORCE CPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import glob
import shutil
import tensorflow as tf
import warnings
from deepface import DeepFace
from datetime import datetime

# --- Suppress TensorFlow Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
JOBS_DIR = "jobs_face"
RESULTS_DIR = "results_face"
PROCESSING_DIR = "jobs_face_processing"

# Points to your "dataset" folder
DB_PATH = "dataset" 

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv" 

# --- SECURITY UPDATE: STRICTNESS CONTROL ---
# Lower number = Stricter (Less false positives, but might miss you if lighting is bad)
# Higher number = Looser (Recognizes you easily, but might mistake friends for you)
# Recommended for Facenet512: 0.30
STRICT_THRESHOLD = 0.30

POLLING_INTERVAL = 0.1 

# --- Ensure directories exist ---
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)

# --- Load Model at Startup ---
print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Starting...")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Loading DeepFace model ({MODEL_NAME}) into memory...")
DeepFace.build_model(MODEL_NAME)
print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Model loaded.")

print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Worker is now running. Watching for jobs in '{JOBS_DIR}'...")

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
            
            processing_path = os.path.join(PROCESSING_DIR, job_filename)
            try:
                shutil.move(job_path, processing_path) 
            except Exception:
                continue 
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] New job received: {job_id}")
            
            try:
                # --- RUN RECOGNITION ---
                dfs = DeepFace.find(
                    img_path=processing_path,
                    db_path=DB_PATH,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    distance_metric="cosine", # We enforce Cosine distance
                    enforce_detection=False, 
                    silent=True
                )
                
                identity = "Unknown"
                
                if len(dfs) > 0 and not dfs[0].empty:
                    # Get the distance score (Lower is better)
                    distance_score = dfs[0].iloc[0]["distance"]
                    
                    # --- SECURITY CHECK ---
                    if distance_score <= STRICT_THRESHOLD:
                        full_path = dfs[0].iloc[0]["identity"]
                        parent_folder = os.path.dirname(full_path)
                        folder_name = os.path.basename(parent_folder)
                        
                        if os.path.abspath(parent_folder) == os.path.abspath(DB_PATH):
                            identity = os.path.basename(full_path).split('.')[0]
                        else:
                            identity = folder_name
                        
                        # Debug print to see how confident the AI was
                        print(f"   >>> Match Found! Distance: {distance_score:.4f} (Limit: {STRICT_THRESHOLD})")
                    else:
                        print(f"   >>> Match Rejected. Distance {distance_score:.4f} is too high (Limit: {STRICT_THRESHOLD})")
                        identity = "Unknown"
                
                with open(result_path, "w") as f:
                    f.write(identity)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Job complete. Result: {identity}")

            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Error processing {job_id}: {e}")
                with open(result_path, "w") as f:
                    f.write("Error")
            
            if os.path.exists(processing_path):
                os.remove(processing_path) 

    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] Shutdown signal received. Exiting.")
        break
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [FACE_WORKER] An unexpected error occurred: {e}")
        time.sleep(1)