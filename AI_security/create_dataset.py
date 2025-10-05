import cv2
import os
import time

# --- Constants ---
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

# --- Initialize Webcam and Face Detector ---
# We now load the XML file from our local project directory
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # <-- THIS LINE IS CHANGED

video_capture = cv2.VideoCapture(0)

print("\nStarting webcam...")
print(f"Preparing to capture {IMAGES_TO_CAPTURE} images.")
print("Please look at the camera and hold still.")
time.sleep(2) # Give user time to prepare

image_count = 0
while image_count < IMAGES_TO_CAPTURE:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale for the face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 1: # We only proceed if exactly one face is detected
        (x, y, w, h) = faces[0]
        
        # Display a green rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing image {image_count + 1}/{IMAGES_TO_CAPTURE}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the captured face
        # We save the whole frame, as face_recognition library works best with it
        image_name = f"{person_name}_{int(time.time())}_{image_count}.jpg"
        image_path = os.path.join(person_path, image_name)
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")
        
        image_count += 1
        time.sleep(CAPTURE_DELAY) # Wait a bit before the next capture

    elif len(faces) > 1:
        cv2.putText(frame, "Multiple faces detected!", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No face detected...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the live feed
    cv2.imshow('Dataset Creator', frame)

    # Allow quitting with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nCaptured {image_count} images successfully.")
print("Dataset creation complete.")

# Release resources
video_capture.release()
cv2.destroyAllWindows()