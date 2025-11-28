import os
import pickle
from deepface import DeepFace
import warnings
import tensorflow as tf

# --- Suppress TensorFlow Warnings ---
# This hides the "TensorFlow Lite XNNPACK" messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) 
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
DATABASE_DIR = "dataset"
REPRESENTATIONS_FILE = "deepface_representations.pkl"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "yolov8"

# --- Logic ---
print(f"--- Starting DeepFace Database Builder ---")
print(f"Loading images from: {DATABASE_DIR}")
print(f"Using Model: {MODEL_NAME}, Detector: {DETECTOR_BACKEND}")
print("This will take a few minutes and will use your GPU...")

representations = []

# Loop over the folders in the dataset
for person_name in os.listdir(DATABASE_DIR):
    person_path = os.path.join(DATABASE_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"\nProcessing folder for: {person_name}")
    # Loop over the images for each person
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        # Check if it's a valid image file
        if not (image_name.lower().endswith('.jpg') or image_name.lower().endswith('.png') or image_name.lower().endswith('.jpeg')):
            continue
            
        try:
            # Use DeepFace to generate the faceprint (embedding)
            # This will automatically find and use your GPU.
            embedding_obj = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                enforce_detection=True,
                detector_backend=DETECTOR_BACKEND
            )
            
            # The 'embedding' is the actual faceprint vector
            embedding = embedding_obj[0]["embedding"]
            
            # Store it with the person's name
            representation_data = [embedding, person_name]
            representations.append(representation_data)
            
            print(f"  [SUCCESS] Processed {image_name}")
            
        except Exception as e:
            # This will catch images where no face was found
            print(f"  [WARNING] Could not process {image_path}: {e}")

# Save the new database to a pickle file
with open(REPRESENTATIONS_FILE, "wb") as f:
    f.write(pickle.dumps(representations))

print(f"\n--- Database creation complete! ---")
print(f"{len(representations)} faceprints saved to {REPRESENTATIONS_FILE}.")