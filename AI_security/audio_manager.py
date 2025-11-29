import time
import os
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import logging

# --- CONFIGURATION ---
CMD_FILE = "audio_cmd.txt"
RESP_FILE = "audio_resp.txt"

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/audio.log', level=logging.INFO, format='[%(asctime)s] %(message)s')

def log(text):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [AUDIO] {text}")
    logging.info(text)

# Init TTS
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    log("TTS Engine Initialized.")
except: log("TTS Engine Failed.")

# Init Mic
recognizer = sr.Recognizer()

def speak(text):
    log(f"Speaking: {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except: pass

def listen():
    with sr.Microphone() as source:
        log("Listening for response...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            log(f"Heard: {text}")
            return text
        except: return None

log("--- AUDIO MANAGER STARTED ---")
if os.path.exists(CMD_FILE): os.remove(CMD_FILE)

while True:
    if os.path.exists(CMD_FILE):
        try:
            with open(CMD_FILE, 'r') as f: cmd = f.read().strip()
            os.remove(CMD_FILE)

            log(f"Command Received: {cmd}")

            if cmd == "WARN_INTRUDER":
                speak("Warning. Intruder detected. Police alerted.")
            elif cmd == "WARN_CRAWLING":
                speak("Suspicious behavior. Stop crawling.")
            elif cmd == "WARN_RUNNING":
                speak("Do not run. Slow down.")
            elif cmd.startswith("WELCOME"):
                name = cmd.split(":")[1]
                speak(f"Welcome home, {name}.")
            elif cmd == "ASK_NAME":
                speak("I do not recognize you. Please state your name.")
                name = listen()
                if name:
                    speak(f"Thank you {name}.")
                    with open(RESP_FILE, "w") as f: f.write(name)
            elif cmd == "FIX_BLUR":
                speak("Please step closer.")
            elif cmd == "WARN_WEAPON":
                speak("Weapon detected! Drop it immediately!")
        except Exception as e:
            log(f"Error: {e}")
    time.sleep(0.2)