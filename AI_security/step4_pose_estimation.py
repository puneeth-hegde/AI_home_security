import face_recognition
import pickle
import cv2
import os
import PySimpleGUI as sg
import mediapipe as mp # New import for pose estimation

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Constants and Setup ---
ENCODINGS_FILE = "encodings.pkl"

# --- Load or Initialize Encodings ---
print("[INFO] Loading encodings...")
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

# --- Webcam Initialization ---
video_capture = cv2.VideoCapture(0)

# --- PySimpleGUI Window Layout ---
sg.theme("DarkBlue")
layout = [
    [sg.Text("AI Security System", size=(40, 1), justification='center', font='Helvetica 20')],
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Text("Press 'e' to enroll. Press 'q' to quit.", key='-TEXT-', font='Helvetica 14')]
]
window = sg.Window('AI Security Cam', layout, location=(800, 400), return_keyboard_events=True, finalize=True)

# --- Main Loop ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from webcam. Exiting...")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- NEW: Process frame for pose estimation ---
    pose_results = pose.process(rgb_frame)
    # Draw the pose annotation on the frame.
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            )

    # --- Face Recognition Logic (remains the same) ---
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = data["names"][first_match_index]
        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # --- PySimpleGUI Event Loop ---
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)
    
    event, values = window.read(timeout=20)
    
    if event == sg.WIN_CLOSED or 'q' in event:
        break
        
    if 'e' in event: # Enrollment logic remains the same
        unknown_faces = [enc for enc in face_encodings if not any(face_recognition.compare_faces(data["encodings"], enc))]
        
        if len(unknown_faces) > 0:
            face_to_enroll_encoding = unknown_faces[0]
            new_name = sg.popup_get_text("An unknown face was detected. Please enter their name:", title="Enroll New Person")

            if new_name and new_name.strip():
                data["encodings"].append(face_to_enroll_encoding)
                data["names"].append(new_name.strip())
                with open(ENCODINGS_FILE, "wb") as f:
                    f.write(pickle.dumps(data))
                print(f"[SUCCESS] {new_name.strip()} has been enrolled.")
                sg.popup(f"{new_name.strip()} has been successfully enrolled!", title="Success")
            else:
                sg.popup("Enrollment cancelled or name was empty.", title="Cancelled")
        else:
            sg.popup("No unknown faces are currently on screen to enroll.", title="Info")

# --- Cleanup ---
video_capture.release()
window.close()