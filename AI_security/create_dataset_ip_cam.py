import cv2
import os
import time

# --- Configuration ---
# --- IMPORTANT: Update this with your DOOR CAMERA's RTSP URL ---
# --- Use stream1 for the highest quality ---
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"

DATASET_DIR = "dataset"
IMAGES_TO_CAPTURE = 10
CAPTURE_DELAY = 0.5 # seconds between captures

# --- Get User's Name ---
person_name = input("Please enter your name: ")
if not person_name:
    print("Name cannot be empty. Exiting.")
    exit()

# --- Create Directories ---
person_path = os.path.join(DATASET_DIR, person_name)

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

if not os.path.exists(person_path):
    os.makedirs(person_path)
    print(f"Created directory: {person_path}")
else:
    print(f"Directory '{person_path}' already exists. Images will be added.")

# --- Initialize Camera and Face Detector ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- THIS SECTION IS UPDATED ---
print(f"Connecting to Door Camera stream: {RTSP_URL_DOOR}")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
video_capture = cv2.VideoCapture(RTSP_URL_DOOR, cv2.CAP_FFMPEG)

if not video_capture.isOpened():
    print("Error: Could not open video stream. Check your RTSP URL and network.")
    exit()
# --- END OF UPDATED SECTION ---

print("\nCamera connected!")
print(f"Preparing to capture {IMAGES_TO_CAPTURE} images.")
print("Please stand in front of the Door Camera and hold still.")
time.sleep(2) # Give user time to prepare

image_count = 0
while image_count < IMAGES_TO_CAPTURE:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing image {image_count + 1}/{IMAGES_TO_CAPTURE}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        image_name = f"{person_name}_{int(time.time())}_{image_count}.jpg"
        image_path = os.path.join(person_path, image_name)
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")
        
        image_count += 1
        time.sleep(CAPTURE_DELAY)

    elif len(faces) > 1:
        cv2.putText(frame, "Multiple faces detected!", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No face detected...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Dataset Creator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nCaptured {image_count} images successfully.")
print("Dataset creation complete.")

# Release resources
video_capture.release()
cv2.destroyAllWindows()