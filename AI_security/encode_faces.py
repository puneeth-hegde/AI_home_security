import face_recognition
import pickle
import os

# Path to the dataset of known faces
dataset_path = "dataset"
encodings_file = "encodings.pkl"

print("[INFO] Starting to process faces...")
known_encodings = []
known_names = []

# Loop over the folders in the dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    # Loop over the images for each person
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        # Load the image and convert it from BGR to RGB
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings for the face in the image
        # We assume each image has only one face
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_encodings.append(encoding)
            known_names.append(person_name)
            print(f"[INFO] Processed {image_name} for {person_name}")

# Save the encodings and names to a file
print("[INFO] Serializing encodings to file...")
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_file, "wb") as f:
    f.write(pickle.dumps(data))
    
print("[INFO] Encodings saved to encodings.pkl")