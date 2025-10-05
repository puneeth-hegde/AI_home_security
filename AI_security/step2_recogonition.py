import face_recognition
import pickle
import cv2

# Load the known faces and encodings
print("[INFO] Loading encodings...")
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Find all faces in the current frame
    face_locations = face_recognition.face_locations(frame)
    # Get face encodings for the faces in the frame
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = data["names"][first_match_index]
        
        # Draw a box around the face
        # Green for known, Red for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()