# audio_manager_v2.py - QUEUE-BASED AUDIO SYSTEM (NO LOST COMMANDS)
import time
import os
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import logging
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
QUEUE_DIR = "audio_queue"
RESP_FILE = "audio_resp.txt"

# Priority levels
PRIORITY_URGENT = 0   # Weapons, intruders
PRIORITY_HIGH = 1     # Running, crawling
PRIORITY_NORMAL = 2   # Welcome, warnings
PRIORITY_LOW = 3      # Info messages

# ============================================================================
# LOGGING SETUP
# ============================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/audio.log', 
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s: %(message)s',
    filemode='a' 
)

def log(text):
    """Logs to both Console and File"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [AUDIO] {text}")
    logging.info(text)

# ============================================================================
# SETUP
# ============================================================================
os.makedirs(QUEUE_DIR, exist_ok=True)

engine = None
recognizer = sr.Recognizer()

def init_engine():
    global engine
    try:
        if engine: 
            engine.stop()
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        log("TTS Engine (Re)Initialized.")
    except Exception as e:
        log(f"CRITICAL: TTS Init Failed: {e}")

# Start engine immediately
init_engine()

def speak(text):
    global engine
    log(f"Speaking: '{text}'")
    try:
        if engine is None: 
            init_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        log(f"TTS Crash: {e}. Restarting engine...")
        init_engine()
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            log("TTS completely failed. Skipping this message.")

def listen():
    with sr.Microphone() as source:
        log("Listening for response...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            text = recognizer.recognize_google(audio)
            log(f"Heard User: '{text}'")
            return text
        except sr.WaitTimeoutError:
            log("No speech detected.")
            return None
        except Exception as e:
            log(f"Mic Error: {e}")
            return None

# ============================================================================
# COMMAND PROCESSOR
# ============================================================================
def process_command(cmd, cmd_file):
    """Process a single command and handle response"""
    log(f"Processing: {cmd}")
    
    try:
        # --- COMMAND HANDLER ---
        if cmd == "WARN_INTRUDER":
            speak("Warning. Intruder detected. Police alerted.")
        
        elif cmd == "WARN_CRAWLING":
            speak("Suspicious behavior. Stand up immediately.")
        
        elif cmd == "WARN_RUNNING":
            speak("Do not run. Slow down.")
        
        elif cmd == "WARN_WEAPON":
            speak("Weapon detected! Drop it immediately.")
        
        elif cmd == "SILENT_ALARM":
            log("ALARM: Silent Trigger Activated.")
            # No audio output for silent alarm
        
        elif cmd.startswith("WELCOME:"):
            name = cmd.split(":", 1)[1]
            # Filter out invalid names
            if name.lower() not in ["unknown", "error", "verifying...", "verifying", ""]:
                speak(f"Welcome home, {name}.")
            else:
                log(f"Blocked invalid welcome for '{name}'")
        
        elif cmd == "ASK_NAME":
            speak("I do not recognize you. Please state your name.")
            name = listen()
            if name:
                speak(f"Thank you, {name}.")
                # Write response for Brain to read
                with open(RESP_FILE, "w") as f: 
                    f.write(name)
            else:
                log("No name provided by visitor")
        
        elif cmd == "FIX_BLUR":
            speak("Please step closer to the camera.")
        
        elif cmd == "FIX_DARK":
            speak("Please turn on the light or step into a brighter area.")
        
        elif cmd == "FIX_BRIGHT":
            speak("Please move away from direct light.")
        
        else:
            log(f"Unknown command: {cmd}")
        
        # Delete command file after processing
        try:
            os.remove(cmd_file)
            log(f"Deleted: {os.path.basename(cmd_file)}")
        except:
            pass
        
        # Brief pause to let audio driver rest
        time.sleep(0.3)
    
    except Exception as e:
        log(f"Command processing error: {e}")
        try:
            os.remove(cmd_file)
        except:
            pass

# ============================================================================
# MAIN LOOP
# ============================================================================
log("=" * 80)
log("AUDIO MANAGER v2.0 - QUEUE-BASED SYSTEM")
log("=" * 80)

# Clear old queue on startup
for old_file in glob.glob(os.path.join(QUEUE_DIR, "*.txt")):
    try:
        os.remove(old_file)
        log(f"Cleared old command: {os.path.basename(old_file)}")
    except:
        pass

log("System ready. Monitoring queue...")

while True:
    try:
        # Get all command files, sorted by priority then timestamp
        cmd_files = glob.glob(os.path.join(QUEUE_DIR, "*.txt"))
        
        if not cmd_files:
            time.sleep(0.1)
            continue
        
        # Sort by filename (priority_timestamp.txt)
        # This ensures urgent commands are processed first
        cmd_files.sort()
        
        # Process the highest priority command
        cmd_file = cmd_files[0]
        
        try:
            with open(cmd_file, 'r') as f:
                cmd = f.read().strip()
            
            if cmd:  # Only process non-empty commands
                process_command(cmd, cmd_file)
            else:
                # Empty file, just delete it
                try:
                    os.remove(cmd_file)
                except:
                    pass
        
        except FileNotFoundError:
            # File was deleted by another process, skip
            pass
        except Exception as e:
            log(f"File read error: {e}")
            try:
                os.remove(cmd_file)
            except:
                pass
    
    except KeyboardInterrupt:
        log("Shutdown requested")
        break
    except Exception as e:
        log(f"Main loop error: {e}")
        time.sleep(0.5)

log("Audio Manager shutting down...")