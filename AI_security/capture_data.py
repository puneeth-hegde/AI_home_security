import cv2
import os
import time

# --- CONFIGURATION ---
RTSP_URL = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(BASE_DIR, "dataset", "puneeth")

os.makedirs(SAVE_FOLDER, exist_ok=True)

# Haar Cascade is fast, but we make it faster by resizing input
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"\n--- FAST DATA COLLECTOR ---")
print(f"Target: {SAVE_FOLDER}")
print(f"Connecting...")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Camera failed.")
    exit()

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.5)
        continue

    # 1. Get Dimensions
    h_orig, w_orig = frame.shape[:2]

    # 2. Create Tiny Frame for Detection (Speed Hack)
    # We detect on 640px wide image instead of 1920px. 3x faster.
    small_frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # 3. Detect Faces on Small Frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 4. Prepare Display
    display_frame = cv2.resize(frame, (1280, 720))
    
    # Calculate Scaling Factors
    # Scale from Small(Detection) to Original(Capture)
    scale_x_orig = w_orig / 640
    scale_y_orig = h_orig / 360
    
    # Scale from Small(Detection) to Display(Screen)
    scale_x_disp = 1280 / 640
    scale_y_disp = 720 / 360

    current_face_crop = None

    # Draw box
    if len(faces) > 0:
        # Take the largest face
        (x, y, w_box, h_box) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        
        # --- Logic for SAVING (High Res) ---
        # Add padding to include hair/chin
        pad = int(h_box * 0.2) 
        
        # Map small coords -> Big Original Coords
        real_x = int(x * scale_x_orig)
        real_y = int(y * scale_y_orig)
        real_w = int(w_box * scale_x_orig)
        real_h = int(h_box * scale_y_orig)
        
        # Calculate Crop Coordinates with padding
        y1 = max(0, real_y - pad)
        y2 = min(h_orig, real_y + real_h + pad)
        x1 = max(0, real_x - pad)
        x2 = min(w_orig, real_x + real_w + pad)
        
        current_face_crop = frame[y1:y2, x1:x2]
        
        # --- Logic for DISPLAY (Screen Res) ---
        disp_x = int(x * scale_x_disp)
        disp_y = int(y * scale_y_disp)
        disp_w = int(w_box * scale_x_disp)
        disp_h = int(h_box * scale_y_disp)
        
        # Draw Green Box on Screen
        cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)

    cv2.putText(display_frame, f"Saved: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Fast Collector - Press S", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        if current_face_crop is not None and current_face_crop.size > 0:
            timestamp = int(time.time())
            filename = f"face_{timestamp}.jpg"
            filepath = os.path.join(SAVE_FOLDER, filename)
            
            # Save the HD Crop
            cv2.imwrite(filepath, current_face_crop)
            print(f"[SAVED] {filename}")
            count += 1
            
            # Visual Flash
            cv2.rectangle(display_frame, (0,0), (1280, 720), (255, 255, 255), 10)
            cv2.imshow("Fast Collector - Press S", display_frame)
            cv2.waitKey(50)
        else:
            print("[ERROR] No face detected! Move closer.")

    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()