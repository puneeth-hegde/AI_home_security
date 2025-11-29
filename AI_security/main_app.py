import cv2
import os
import time
import threading
import queue
import numpy as np
import glob
import torch
from ultralytics import YOLO
from sort import Sort
from collections import deque, Counter

RTSP_URL_GATE = "rtsp://Gate_Camera:CREAKmyPASSWORD1219!!!@192.168.0.115:554/stream1"
RTSP_URL_DOOR = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.0.120:554/stream1"
YOLO_MODEL = "yolov8n.pt"

AI_W, AI_H = 640, 360
DISP_W, DISP_H = 960, 540
WEAPON_CLASSES = [43, 76, 34]
WEAPON_CONF = 0.20

RUNNING_AREA_GROWTH = 1.5
RUNNING_SPEED_THRESHOLD = 80
IDLE_FRAMES_BEFORE_ASK = 30
RECOGNITION_VOTE_REQUIRED = 3
IDENTITY_LOCK_TIMEOUT = 10  # seconds - keep identity locked even if face turns away

CMD_FILE = "audio_cmd.txt"
RESP_FILE = "audio_resp.txt"
JOBS_FACE_DIR = "jobs_face"
JOBS_POSE_DIR = "jobs_pose"
RESULTS_FACE_DIR = "results_face"
RESULTS_POSE_DIR = "results_pose"

for d in [JOBS_FACE_DIR, JOBS_POSE_DIR, RESULTS_FACE_DIR, RESULTS_POSE_DIR]:
    os.makedirs(d, exist_ok=True)

model = YOLO(YOLO_MODEL)
if torch.cuda.is_available():
    model.to('cuda')
tracker = Sort(max_age=30, min_hits=3)

class PersonState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.identity = "Unknown"
        self.identity_votes = deque(maxlen=5)
        self.identity_locked = False
        self.identity_lock_time = 0  # when identity was locked
        self.bbox_history = deque(maxlen=10)
        self.centroid_history = deque(maxlen=10)
        self.behavior = "IDLE"
        self.idle_frames = 0
        self.name_asked = False
        self.welcomed = False
        self.pose_status = "NORMAL"
        self.has_weapon = False
        self.last_face_check = 0
        self.last_pose_check = 0
        self.face_quality = "OK"
        
    def update_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.bbox_history.append((bbox, area))
        self.centroid_history.append((cx, cy))
        
        if len(self.bbox_history) >= 10:
            old_area = self.bbox_history[0][1]
            new_area = self.bbox_history[-1][1]
            area_growth = new_area / (old_area + 1e-6)
            old_cx, old_cy = self.centroid_history[0]
            new_cx, new_cy = self.centroid_history[-1]
            distance = np.sqrt((new_cx - old_cx)**2 + (new_cy - old_cy)**2)
            speed = distance / 10
            
            if speed > RUNNING_SPEED_THRESHOLD or area_growth > RUNNING_AREA_GROWTH:
                self.behavior = "RUNNING"
                self.idle_frames = 0
            else:
                self.behavior = "IDLE"
                self.idle_frames += 1
        else:
            self.behavior = "IDLE"
            self.idle_frames += 1
    
    def vote_identity(self, name):
        self.identity_votes.append(name)
        
        # If already locked and not timed out, keep current identity
        if self.identity_locked and (time.time() - self.identity_lock_time) < IDENTITY_LOCK_TIMEOUT:
            if name == self.identity:  # Reinforce if same person
                return True
            # Ignore conflicting votes during lock period
            return False
        
        # Voting logic
        if len(self.identity_votes) >= RECOGNITION_VOTE_REQUIRED:
            votes = Counter(self.identity_votes)
            top_name, top_count = votes.most_common(1)[0]
            if top_count >= RECOGNITION_VOTE_REQUIRED:
                self.identity = top_name
                self.identity_locked = True
                self.identity_lock_time = time.time()
                return True
        return False
    
    def get_color(self):
        if self.has_weapon or self.pose_status == "VIOLENCE":
            return (0, 0, 255)
        elif self.behavior == "RUNNING":
            return (0, 140, 255)
        elif self.identity == "puneeth" and self.pose_status == "SURRENDER":
            return (0, 0, 255)
        elif self.identity == "puneeth":
            return (0, 255, 0)
        else:
            return (255, 255, 0)

person_memory = {}
frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

def capture_thread(url, name):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            try:
                frame_queue.put((name, frame), timeout=0.1)
            except queue.Full:
                pass
        else:
            time.sleep(0.1)
    cap.release()

def send_audio_command(cmd):
    try:
        with open(CMD_FILE, 'w') as f:
            f.write(cmd)
    except: pass

def check_audio_response():
    if os.path.exists(RESP_FILE):
        try:
            with open(RESP_FILE, 'r') as f:
                name = f.read().strip()
            os.remove(RESP_FILE)
            return name
        except: pass
    return None

def dispatch_face_job(track_id, face_crop):
    job_path = os.path.join(JOBS_FACE_DIR, f"{track_id}_{int(time.time()*1000)}.jpg")
    cv2.imwrite(job_path, face_crop)

def dispatch_pose_job(track_id, person_crop):
    job_path = os.path.join(JOBS_POSE_DIR, f"{track_id}_{int(time.time()*1000)}.jpg")
    cv2.imwrite(job_path, person_crop)

def check_worker_results():
    for result_file in glob.glob(os.path.join(RESULTS_FACE_DIR, "*.txt")):
        try:
            with open(result_file, 'r') as f:
                lines = f.read().strip().split('\n')
            track_id = int(lines[0].split(':')[1])
            identity = lines[1].split(':')[1]
            quality = lines[2].split(':')[1]
            if track_id in person_memory:
                person_memory[track_id].vote_identity(identity)
                person_memory[track_id].face_quality = quality
            os.remove(result_file)
        except: pass
    
    for result_file in glob.glob(os.path.join(RESULTS_POSE_DIR, "*.txt")):
        try:
            with open(result_file, 'r') as f:
                lines = f.read().strip().split('\n')
            track_id = int(lines[0].split(':')[1])
            pose_status = lines[1].split(':')[1]
            if track_id in person_memory:
                person_memory[track_id].pose_status = pose_status
            os.remove(result_file)
        except: pass

def main():
    # Start BOTH camera threads
    threading.Thread(target=capture_thread, args=(RTSP_URL_DOOR, "door"), daemon=True).start()
    threading.Thread(target=capture_thread, args=(RTSP_URL_GATE, "gate"), daemon=True).start()
    
    cv2.namedWindow("Door Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gate Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Door Camera", DISP_W, DISP_H)
    cv2.resizeWindow("Gate Camera", DISP_W, DISP_H)
    
    door_frame = None
    gate_frame = None
    
    while True:
        # Get frames from BOTH cameras
        try:
            camera_name, frame = frame_queue.get(timeout=0.01)
            if camera_name == "door":
                door_frame = frame.copy()
            elif camera_name == "gate":
                gate_frame = frame.copy()
        except queue.Empty:
            pass
        
        # Process DOOR camera (main recognition)
        if door_frame is not None:
            processed_door = process_frame(door_frame, "DOOR")
            cv2.imshow("Door Camera", processed_door)
        
        # Process GATE camera (early warning)
        if gate_frame is not None:
            processed_gate = process_frame(gate_frame, "GATE")
            cv2.imshow("Gate Camera", processed_gate)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    stop_event.set()
    cv2.destroyAllWindows()

def process_frame(frame, cam_name):
def process_frame(frame, cam_name):
    """Process a single frame from either camera"""
    h, w = frame.shape[:2]
    frame_ai = cv2.resize(frame, (AI_W, AI_H))
    results = model(frame_ai, verbose=False, conf=0.25)
    
    detections = []
    weapons_detected = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = int(x1*w/AI_W), int(y1*h/AI_H), int(x2*w/AI_W), int(y2*h/AI_H)
            
            if cls == 0:
                detections.append([x1, y1, x2, y2, conf])
            elif cls in WEAPON_CLASSES and conf > WEAPON_CONF:
                weapons_detected.append((x1, y1, x2, y2))
    
    tracks = tracker.update(np.array(detections) if len(detections) > 0 else np.empty((0, 5)))
    current_ids = set()
    
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        current_ids.add(track_id)
        
        if track_id not in person_memory:
            person_memory[track_id] = PersonState(track_id)
        
        person = person_memory[track_id]
        person.update_bbox([x1, y1, x2, y2])
        
        person_center = ((x1+x2)/2, (y1+y2)/2)
        person.has_weapon = any(wx1<person_center[0]<wx2 and wy1<person_center[1]<wy2 for wx1,wy1,wx2,wy2 in weapons_detected)
        
        # Only do face/pose on DOOR camera
        if cam_name == "DOOR":
            if time.time() - person.last_face_check > 0.5:
                face_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if face_crop.size > 0:
                    dispatch_face_job(track_id, face_crop)
                    person.last_face_check = time.time()
            
            if time.time() - person.last_pose_check > 0.3:
                person_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if person_crop.size > 0:
                    dispatch_pose_job(track_id, person_crop)
                    person.last_pose_check = time.time()
        
        if person.behavior == "RUNNING" and person.identity != "puneeth":
            send_audio_command("WARN_INTRUDER")
        if person.has_weapon:
            send_audio_command("WARN_WEAPON")
        if person.identity == "puneeth" and person.pose_status == "SURRENDER":
            send_audio_command("WARN_INTRUDER")
        if person.identity == "puneeth" and not person.welcomed:
            send_audio_command(f"WELCOME:puneeth")
            person.welcomed = True
        if (person.behavior == "IDLE" and person.identity == "Unknown" and 
            person.idle_frames > IDLE_FRAMES_BEFORE_ASK and not person.name_asked and person.identity_locked):
            send_audio_command("ASK_NAME")
            person.name_asked = True
            visitor_name = check_audio_response()
            if visitor_name:
                person.identity = visitor_name
        if person.face_quality == "BLUR":
            send_audio_command("FIX_BLUR")
        
        color = person.get_color()
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"ID:{track_id} {person.identity} [{person.behavior}]"
        if person.has_weapon: label += " WEAPON!"
        if person.pose_status == "SURRENDER": label += " HANDS UP"
        cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    for tid in list(person_memory.keys()):
        if tid not in current_ids:
            del person_memory[tid]
    
    check_worker_results()
    frame_display = cv2.resize(frame, (DISP_W, DISP_H))
    cv2.putText(frame_display, f"{cam_name}: {len(current_ids)} persons", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return frame_display

if __name__ == "__main__":
    main()