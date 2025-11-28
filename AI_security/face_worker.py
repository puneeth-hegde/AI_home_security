import os
import time
import glob
import cv2
import numpy as np
from deepface import DeepFace

# Configuration
JOBS_DIR = "jobs_face"
RESULTS_DIR = "results_face"
DB_PATH = "dataset"
MODEL_NAME = "Facenet512"

# Multi-tier thresholding for steep angles
THRESHOLD_STRICT = 0.20      # High-quality frontal faces
THRESHOLD_MODERATE = 0.28    # Slightly angled faces
THRESHOLD_RELAXED = 0.35     # Steep angle faces (only with quality checks)

# Quality thresholds
MIN_FACE_SIZE = 80
MIN_BLUR_SCORE = 100
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

# Face detection backends (fallback order)
# OpenCV is instant. Retinaface is slow.
DETECTION_BACKENDS = ['opencv', 'ssd', 'mtcnn', 'retinaface']

# Build model once at startup
print("Loading face recognition model...")
DeepFace.build_model(MODEL_NAME)
print("Model loaded successfully")

def read_metadata(job_id):
    """Read metadata file if it exists"""
    meta_path = os.path.join(JOBS_DIR, f"{job_id}_meta.txt")
    metadata = {}
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.strip().split(':', 1)
                        try:
                            metadata[key] = float(val) if '.' in val else int(val)
                        except:
                            metadata[key] = val
        except:
            pass
    
    return metadata

def enhance_image_quality(img):
    """Apply preprocessing to improve face detection on steep angles"""
    # Convert to LAB color space for better lighting normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Slight sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend original with sharpened
    result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    return result

def detect_face_with_fallback(img_path):
    """Try multiple detection backends to find face"""
    img = cv2.imread(img_path)
    
    # Try enhanced preprocessing
    enhanced = enhance_image_quality(img)
    cv2.imwrite(img_path.replace('.jpg', '_enhanced.jpg'), enhanced)
    
    for backend in DETECTION_BACKENDS:
        try:
            # Try with enhanced image first
            result = DeepFace.extract_faces(
                img_path=img_path.replace('.jpg', '_enhanced.jpg'),
                detector_backend=backend,
                enforce_detection=True,
                align=True
            )
            
            if result and len(result) > 0:
                # Get the largest face
                largest = max(result, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                
                # Clean up enhanced image
                try:
                    os.remove(img_path.replace('.jpg', '_enhanced.jpg'))
                except:
                    pass
                
                return largest, backend
        except:
            # Try original image
            try:
                result = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True
                )
                
                if result and len(result) > 0:
                    largest = max(result, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                    return largest, backend
            except:
                continue
    
    # Clean up enhanced image if exists
    try:
        os.remove(img_path.replace('.jpg', '_enhanced.jpg'))
    except:
        pass
    
    return None, None

def calculate_face_confidence(face_data, metadata):
    """Calculate confidence score based on face detection quality"""
    score = 0.0
    
    if not face_data:
        return score
    
    # Confidence from detector
    if 'confidence' in face_data:
        score += face_data['confidence'] * 30
    
    # Size score
    area = face_data['facial_area']
    face_size = area['w'] * area['h']
    if face_size > 10000:
        score += 25
    elif face_size > 5000:
        score += 15
    elif face_size > 2500:
        score += 5
    
    # Quality from metadata
    if metadata.get('blur_score', 0) > MIN_BLUR_SCORE:
        score += 25
    elif metadata.get('blur_score', 0) > MIN_BLUR_SCORE / 2:
        score += 10
    
    # Brightness
    brightness = metadata.get('brightness', 0)
    if MIN_BRIGHTNESS < brightness < MAX_BRIGHTNESS:
        score += 20
    elif brightness > 0:
        score += 5
    
    return min(score, 100)

def adaptive_recognize(img_path, metadata):
    """
    Multi-stage recognition with adaptive thresholding
    Returns: (identity, confidence, method_used)
    """
    
    # Stage 1: Try strict detection with enforce_detection=True
    face_data, backend = detect_face_with_fallback(img_path)
    
    if face_data:
        # Face detected - calculate quality score
        quality_score = calculate_face_confidence(face_data, metadata)
        
        # Choose threshold based on quality
        if quality_score >= 70:
            threshold = THRESHOLD_STRICT
            method = "STRICT"
        elif quality_score >= 40:
            threshold = THRESHOLD_MODERATE
            method = "MODERATE"
        else:
            threshold = THRESHOLD_RELAXED
            method = "RELAXED"
        
        # Perform recognition with enforce_detection=True
        try:
            dfs = DeepFace.find(
                img_path=img_path,
                db_path=DB_PATH,
                model_name=MODEL_NAME,
                detector_backend=backend,
                distance_metric="cosine",
                enforce_detection=True,
                silent=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                distance = dfs[0].iloc[0]["distance"]
                
                if distance <= threshold:
                    identity = os.path.basename(os.path.dirname(dfs[0].iloc[0]["identity"]))
                    confidence = 1 - distance
                    return identity, confidence, method
        except:
            pass
    
    # Stage 2: Face not detected or no match - try with enforce_detection=False
    # BUT with very strict threshold to avoid false positives
    try:
        # Preprocess image
        img = cv2.imread(img_path)
        enhanced = enhance_image_quality(img)
        temp_path = img_path.replace('.jpg', '_temp.jpg')
        cv2.imwrite(temp_path, enhanced)
        
        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend='opencv',
            distance_metric="cosine",
            enforce_detection=False,
            silent=True
        )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if len(dfs) > 0 and not dfs[0].empty:
            distance = dfs[0].iloc[0]["distance"]
            
            # VERY strict threshold when enforce_detection=False
            # Only accept if distance is extremely low
            if distance <= 0.18:  # Much stricter than normal
                identity = os.path.basename(os.path.dirname(dfs[0].iloc[0]["identity"]))
                confidence = 1 - distance
                return identity, confidence, "FALLBACK-STRICT"
    except Exception as e:
        print(f"Fallback recognition error: {e}")
    
    # Stage 3: No match found
    return "Unknown", 0.0, "NO_MATCH"

def process_job(job_path):
    """Process a single face recognition job"""
    job_id = os.path.basename(job_path).split('.')[0]
    
    try:
        # Read metadata
        metadata = read_metadata(job_id)
        
        # Check basic quality requirements
        if metadata:
            blur = metadata.get('blur_score', float('inf'))
            brightness = metadata.get('brightness', 128)
            size_ok = metadata.get('size_ok', True)
            
            # Reject very poor quality images immediately
            if not size_ok or blur < 50 or brightness < 20 or brightness > 240:
                return "Unknown", 0.0, "POOR_QUALITY"
        
        # Adaptive recognition
        identity, confidence, method = adaptive_recognize(img_path, metadata)
        
        # Format result
        if identity != "Unknown":
            result = f"{identity}|{confidence:.3f}|{method}"
        else:
            result = "Unknown"
        
        return result, confidence, method
    
    except Exception as e:
        print(f"Error processing {job_id}: {e}")
        return "Error", 0.0, "ERROR"

# Main worker loop
print("Face recognition worker started")
print(f"Database: {DB_PATH}")
print(f"Thresholds - Strict: {THRESHOLD_STRICT}, Moderate: {THRESHOLD_MODERATE}, Relaxed: {THRESHOLD_RELAXED}")

while True:
    job_files = glob.glob(os.path.join(JOBS_DIR, "*.jpg"))
    
    if not job_files:
        time.sleep(0.05)
        continue
    
    for job_path in job_files:
        job_id = os.path.basename(job_path).split('.')[0]
        
        # Process the job
        result, confidence, method = process_job(job_path)
        
        # Write result
        try:
            result_path = os.path.join(RESULTS_DIR, f"result_{job_id}.txt")
            with open(result_path, "w") as f:
                f.write(result)
            
            # Log for debugging
            if result != "Error":
                print(f"{job_id}: {result} (Method: {method})")
        except Exception as e:
            print(f"Error writing result: {e}")
        
        # Clean up job file and metadata
        try:
            os.remove(job_path)
            meta_path = os.path.join(JOBS_DIR, f"{job_id}_meta.txt")
            if os.path.exists(meta_path):
                os.remove(meta_path)
        except:
            pass