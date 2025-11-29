# face_worker_v2.py - FIXED FACE RECOGNITION WITH SMART VOTING
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
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
JOBS_DIR = "jobs_face"
RESULTS_DIR = "results_face"
DB_PATH = "dataset"
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"

# QUALITY THRESHOLDS
BLUR_VAR_THRESHOLD = 100.0      # Increased from 80
BRIGHTNESS_MIN = 50.0           # Increased from 45
BRIGHTNESS_MAX = 230.0          # Too bright = washed out
MIN_FACE_SIZE = 80              # Increased from 60

# RECOGNITION THRESHOLDS (More conservative)
GOOD_MATCH_THRESHOLD = 0.20     # High confidence match
ACCEPTABLE_MATCH_THRESHOLD = 0.30  # Medium confidence (needs more votes)
DOOR_CAMERA_BOOST = 0.05        # Add this to threshold for door camera

# VOTING SYSTEM (Quality-weighted)
VOTE_HISTORY_SIZE = 8           # Keep last 8 frames
VOTES_REQUIRED_HIGH_QUAL = 3    # If quality is good
VOTES_REQUIRED_LOW_QUAL = 5     # If quality is borderline

# ============================================================================
# LOGGING SETUP
# ============================================================================
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

logger.info("=" * 80)
logger.info("FACE RECOGNITION ENGINE v2.0 - STARTING")
logger.info("=" * 80)

# Load model once
logger.info("Loading DeepFace Model...")
DeepFace.build_model(MODEL_NAME)
logger.info("Model loaded successfully")

# ============================================================================
# VOTE MEMORY (Per-Person Tracking)
# ============================================================================
# Structure: {person_id: deque([{'name': 'puneeth', 'dist': 0.18, 'quality': 0.95}, ...])}
vote_memory = {}

# ============================================================================
# QUALITY ANALYSIS
# ============================================================================
def variance_of_laplacian(gray):
    """Measure image sharpness (blur detection)"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_quality(img, job_id):
    """
    Returns: (is_valid, error_code, quality_score, metadata)
    quality_score: 0.0-1.0 where 1.0 is perfect
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    metadata = {}
    quality_score = 1.0  # Start perfect, deduct for issues
    
    # 1. SIZE CHECK
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        logger.info(f"{job_id} REJECTED: TOO_SMALL size={w}x{h}")
        return False, "ERROR:TOO_SMALL", 0.0, {"size": f"{w}x{h}"}
    
    # 2. BLUR CHECK
    blur_var = variance_of_laplacian(gray)
    metadata['blur_var'] = round(blur_var, 2)
    
    if blur_var < BLUR_VAR_THRESHOLD:
        logger.info(f"{job_id} REJECTED: BLURRY blur_var={blur_var:.1f}")
        return False, "ERROR:BLURRY", 0.0, metadata
    elif blur_var < BLUR_VAR_THRESHOLD * 1.3:
        quality_score -= 0.2  # Borderline blur
        logger.info(f"{job_id} WARNING: Borderline blur={blur_var:.1f}")
    
    # 3. BRIGHTNESS CHECK
    mean_brightness = float(np.mean(gray))
    metadata['brightness'] = round(mean_brightness, 1)
    
    if mean_brightness < BRIGHTNESS_MIN:
        logger.info(f"{job_id} REJECTED: TOO_DARK brightness={mean_brightness:.1f}")
        return False, "ERROR:DARK", 0.0, metadata
    elif mean_brightness > BRIGHTNESS_MAX:
        logger.info(f"{job_id} REJECTED: TOO_BRIGHT brightness={mean_brightness:.1f}")
        return False, "ERROR:BRIGHT", 0.0, metadata
    elif mean_brightness < BRIGHTNESS_MIN * 1.2:
        quality_score -= 0.15  # Slightly dark
    elif mean_brightness > BRIGHTNESS_MAX * 0.9:
        quality_score -= 0.15  # Slightly bright
    
    # 4. CONTRAST CHECK
    contrast = gray.std()
    metadata['contrast'] = round(contrast, 2)
    
    if contrast < 20:
        logger.info(f"{job_id} REJECTED: LOW_CONTRAST contrast={contrast:.1f}")
        return False, "ERROR:FLAT", 0.0, metadata
    elif contrast < 30:
        quality_score -= 0.1
    
    # 5. EDGE DENSITY (Anti-spoofing)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    metadata['edge_density'] = round(edge_density, 3)
    
    if edge_density < 0.03 or edge_density > 0.30:
        logger.warning(f"{job_id} SUSPECTED_SPOOF: edge_density={edge_density:.3f}")
        return False, "ERROR:SPOOF:TEXTURE", 0.0, metadata
    
    # 6. FACE ANGLE ESTIMATION (Using aspect ratio as proxy)
    aspect_ratio = w / h
    metadata['aspect_ratio'] = round(aspect_ratio, 2)
    
    # Normal face: 0.7-0.9 aspect ratio
    # Tilted face: <0.6 or >1.1
    if aspect_ratio < 0.6 or aspect_ratio > 1.1:
        quality_score -= 0.25
        logger.info(f"{job_id} WARNING: Unusual angle aspect={aspect_ratio:.2f}")
    
    quality_score = max(0.0, min(1.0, quality_score))  # Clamp to 0-1
    metadata['quality_score'] = round(quality_score, 2)
    
    logger.info(f"{job_id} QUALITY_OK: score={quality_score:.2f} blur={blur_var:.1f} brightness={mean_brightness:.1f}")
    return True, "OK", quality_score, metadata

# ============================================================================
# FACE RECOGNITION WITH SMART VOTING
# ============================================================================
def recognize_face(img, job_path, job_id, quality_score):
    """
    Returns: final_identity (string)
    """
    # Determine if this is a door camera (top-down angle)
    is_door_camera = 'door' in job_id.lower()
    
    # Adjust threshold based on camera type
    base_threshold = GOOD_MATCH_THRESHOLD
    if is_door_camera:
        threshold = base_threshold + DOOR_CAMERA_BOOST
        logger.info(f"{job_id} Using DOOR camera threshold={threshold:.2f}")
    else:
        threshold = base_threshold
    
    try:
        # Run DeepFace recognition
        dfs = DeepFace.find(
            img_path=job_path, 
            db_path=DB_PATH, 
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND, 
            distance_metric="cosine",
            enforce_detection=False, 
            silent=True
        )
        
        raw_identity = "Unknown"
        distance = 1.0
        
        # Parse results
        if len(dfs) > 0 and not dfs[0].empty:
            distance = float(dfs[0].iloc[0]["distance"])
            
            # Only accept if below threshold
            if distance <= ACCEPTABLE_MATCH_THRESHOLD:
                path = dfs[0].iloc[0]["identity"]
                if os.path.sep in path:
                    raw_identity = os.path.basename(os.path.dirname(path))
                else:
                    raw_identity = os.path.basename(path).split('.')[0]
            
            logger.info(f"{job_id} DeepFace result: identity={raw_identity} distance={distance:.3f} threshold={threshold:.2f}")
        else:
            logger.info(f"{job_id} No faces detected by DeepFace")
        
        # ====================================================================
        # VOTING SYSTEM (Quality-Weighted)
        # ====================================================================
        
        # Initialize vote history for this person
        if job_id not in vote_memory:
            vote_memory[job_id] = deque(maxlen=VOTE_HISTORY_SIZE)
            logger.info(f"{job_id} Initialized vote memory")
        
        # Add current vote
        vote_memory[job_id].append({
            'name': raw_identity,
            'distance': distance,
            'quality': quality_score,
            'timestamp': time.time()
        })
        
        # Count votes
        vote_counts = Counter([v['name'] for v in vote_memory[job_id]])
        most_common_name, vote_count = vote_counts.most_common(1)[0]
        
        # Calculate average quality of votes for most common name
        votes_for_winner = [v for v in vote_memory[job_id] if v['name'] == most_common_name]
        avg_quality = sum([v['quality'] for v in votes_for_winner]) / len(votes_for_winner)
        avg_distance = sum([v['distance'] for v in votes_for_winner]) / len(votes_for_winner)
        
        # Determine votes required based on quality
        if avg_quality >= 0.8:
            votes_needed = VOTES_REQUIRED_HIGH_QUAL
        else:
            votes_needed = VOTES_REQUIRED_LOW_QUAL
        
        logger.info(f"{job_id} VOTES: {most_common_name}={vote_count}/{len(vote_memory[job_id])} " +
                   f"needed={votes_needed} avg_quality={avg_quality:.2f} avg_dist={avg_distance:.3f}")
        
        # ====================================================================
        # DECISION LOGIC
        # ====================================================================
        
        # Case 1: High-confidence match with enough votes
        if vote_count >= votes_needed and avg_distance <= threshold:
            final_identity = most_common_name
            logger.info(f"{job_id} ✓ VERIFIED: {final_identity}")
        
        # Case 2: Not enough votes yet
        elif vote_count < votes_needed:
            final_identity = "Verifying..."
            logger.info(f"{job_id} PENDING: Need {votes_needed - vote_count} more consistent frames")
        
        # Case 3: Consistent "Unknown" votes
        elif most_common_name == "Unknown" and vote_count >= votes_needed:
            final_identity = "Unknown"
            logger.info(f"{job_id} CONFIRMED: Unknown person")
        
        # Case 4: Distance too high even with votes
        else:
            final_identity = "Unknown"
            logger.info(f"{job_id} REJECTED: Distance too high ({avg_distance:.3f} > {threshold:.2f})")
        
        return final_identity
    
    except Exception as e:
        logger.error(f"{job_id} RECOGNITION_ERROR: {e}")
        return "Error"

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
logger.info("Entering main processing loop...")
processed_count = 0
last_cleanup = time.time()

while True:
    try:
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
        
        if not job_files:
            time.sleep(0.05)
            continue
        
        for job_path in job_files:
            job_filename = os.path.basename(job_path)
            job_id = job_filename.split('.')[0]  # e.g., "door_7" or "gate_3"
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            
            try:
                # Read image
                img = cv2.imread(job_path)
                if img is None:
                    logger.error(f"{job_id} INVALID_IMAGE")
                    with open(result_path, "w") as f:
                        f.write("Error")
                    os.remove(job_path)
                    continue
                
                # Quality analysis
                is_valid, error_code, quality_score, quality_meta = analyze_quality(img, job_id)
                
                if not is_valid:
                    # Write error code as result
                    with open(result_path, "w") as f:
                        f.write(error_code)
                    try:
                        os.remove(job_path)
                    except:
                        pass
                    continue
                
                # Face recognition with voting
                identity = recognize_face(img, job_path, job_id, quality_score)
                
                # Write result
                with open(result_path, "w") as f:
                    f.write(identity)
                
                processed_count += 1
                
                # Log milestone
                if processed_count % 50 == 0:
                    logger.info(f"MILESTONE: {processed_count} faces processed")
            
            except Exception as e:
                logger.error(f"{job_id} EXCEPTION: {e}")
                with open(result_path, "w") as f:
                    f.write("Error")
            
            # Cleanup job file
            try:
                os.remove(job_path)
            except:
                pass
        
        # Periodic cleanup of old vote memory (every 30 seconds)
        if time.time() - last_cleanup > 30:
            current_time = time.time()
            stale_ids = []
            
            for person_id, votes in vote_memory.items():
                if votes:
                    last_vote_time = votes[-1]['timestamp']
                    if current_time - last_vote_time > 60:  # 60 seconds inactive
                        stale_ids.append(person_id)
            
            for person_id in stale_ids:
                logger.info(f"Clearing stale vote memory for {person_id}")
                del vote_memory[person_id]
            
            last_cleanup = current_time
    
    except KeyboardInterrupt:
        logger.info("SHUTDOWN_REQUESTED")
        break
    except Exception as e:
        logger.error(f"MAIN_LOOP_EXCEPTION: {e}")
        time.sleep(0.1)

logger.info(f"SHUTDOWN: {processed_count} total faces processed")