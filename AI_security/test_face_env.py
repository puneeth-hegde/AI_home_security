import cv2
import os
import pickle
import time
from deepface import DeepFace
import warnings
import tensorflow as tf

# --- Suppress TensorFlow Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
# --- IMPORTANT: Update this with your DOOR CAMERA's RTSP URL ---
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
REPRESENTATIONS_FILE = "deepface_representations.pkl"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "yolov8"
PROCESS_EVERY_N_FRAMES = 10 

# --- Load Database ---
print(f"[INFO] Loading face database: {REPRESENTATIONS_FILE}...")
with open(REPRESENTATIONS_FILE, "rb") as f:
    database = pickle.load(f)
print(f"[INFO] Database loaded with {len(database)} known faces.")
print("[INFO] Loading DeepFace model. This may take a moment...")
DeepFace.build_model(MODEL_NAME)
print("[INFO] Model loaded.")

# --- Connect to Camera ---
print(f"Connecting to Door Camera stream: {RTSP_URL_DOOR}")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
video_capture = cv2.VideoCapture(RTSP_URL_DOOR, cv2.CAP_FFMPEG)
if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()
print("[INFO] Camera connected! Starting test loop...")

frame_counter = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    frame_counter += 1
    
    # --- Run AI only on specified frames ---
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        try:
            faces_info = DeepFace.extract_faces(
                img_path=frame, 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=False
            )
            
            for i, face_info in enumerate(faces_info):
                if face_info['confidence'] == 0: continue

                face_img = face_info['face']
                
                # --- THIS IS THE CORRECTED CODE ---
                facial_area = face_info['facial_area']
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                # --- END OF FIX ---

                found_name = "Unknown"
                
                for db_embedding, db_name in database:
                    result = DeepFace.verify(
                        img1_path=face_img, 
                        img2_path=db_embedding,
                        model_name=MODEL_NAME, 
                        enforce_detection=False,
                        detector_backend='skip',
                        silent=True
                    )
                    
                    if result["verified"]:
                        found_name = db_name
                        break 
                
                color = (0, 255, 0) if found_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, found_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except Exception as e:
            # We will print the error but not crash
            print(f"[ERROR] Could not process frame: {e}")

    # Display the frame (will be laggy, this is expected)
    cv2.imshow("Face Recognition Test (Expect Lag)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Test complete.")