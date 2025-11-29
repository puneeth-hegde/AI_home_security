# audio_manager.py - OPTIMIZED AUDIO HANDLER
import time
import os
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================
CMD_FILE = "audio_cmd.txt"
RESP_FILE = "audio_resp.txt"

# ============================================================================
# LOGGING
# ============================================================================
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger('audio_manager')
handler = logging.FileHandler('logs/audio_manager.log')
handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setFormatter(logging.Formatter('[%(asctime)s] [AUDIO] %(message)s'))
logger.addHandler(console)

# ============================================================================
# TTS INITIALIZATION
# ============================================================================
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    logger.info("TTS Engine Ready")
except Exception as e:
    logger.error(f"TTS Init Error: {e}")
    engine = None

# ============================================================================
# STT INITIALIZATION
# ============================================================================
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8

logger.info("Available Microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    logger.info(f"  {index}: {name}")

LAPTOP_MIC_INDEX = None  # Set to specific index if needed

def get_microphone():
    if LAPTOP_MIC_INDEX is not None:
        return sr.Microphone(device_index=LAPTOP_MIC_INDEX)
    else:
        return sr.Microphone()

# ============================================================================
# AUDIO FUNCTIONS
# ============================================================================
def speak(text):
    """Text-to-speech with logging"""
    logger.info(f"Speaking: {text}")
    
    if engine is None:
        logger.error("TTS engine not available")
        return
    
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS Error: {e}")

def listen():
    """Speech-to-text with timeout"""
    try:
        with get_microphone() as source:
            logger.info("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info(f"Energy threshold: {recognizer.energy_threshold}")
            
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            
            logger.info(f"Heard: {text}")
            return text
    
    except sr.WaitTimeoutError:
        logger.warning("Timeout - no speech detected")
        return None
    
    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
        return None
    
    except sr.RequestError as e:
        logger.error(f"API Error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Listen Error: {e}")
        return None

# ============================================================================
# COMMAND PROCESSOR
# ============================================================================
def process_command(cmd):
    """Handle all audio commands"""
    logger.info(f"Processing command: {cmd}")
    
    if cmd == "WARN_INTRUDER":
        speak("Warning. Intruder detected. Alert triggered.")
    
    elif cmd.startswith("WELCOME:"):
        name = cmd.split(':', 1)[1]
        speak(f"Welcome home, {name}.")
    
    elif cmd == "ASK_NAME":
        speak("I do not recognize you. Please state your name clearly.")
        name = listen()
        
        if name:
            try:
                with open(RESP_FILE, "w") as f:
                    f.write(name)
                logger.info(f"Saved name response: {name}")
            except Exception as e:
                logger.error(f"Write RESP Error: {e}")
            
            speak(f"Thank you, {name}. You are now logged.")
        else:
            speak("I could not hear your name. Please try again later.")
    
    elif cmd == "FIX_BLUR":
        speak("Your face is not clear. Please step closer to the camera.")
    
    elif cmd == "FIX_DARK":
        speak("The lighting is too dark. Please turn on a light.")
    
    elif cmd == "WARN_WEAPON":
        speak("Warning. Weapon detected. Security alert.")
    
    elif cmd == "WARN_RUNNING":
        speak("Stop running. Please remain still.")
    
    elif cmd == "SILENT_ALARM":
        # Silent alarm (no audio, just log)
        logger.warning("SILENT ALARM TRIGGERED - Hostage situation detected")
        # In production: send notification to phone, log to security service, etc.
    
    else:
        logger.warning(f"Unknown command: {cmd}")

# ============================================================================
# MAIN LOOP
# ============================================================================
logger.info("Audio Manager Started - Monitoring for commands...")
processed_count = 0

while True:
    if os.path.exists(CMD_FILE):
        try:
            with open(CMD_FILE, 'r') as f:
                cmd = f.read().strip()
            
            # Remove command file immediately to avoid re-processing
            try:
                os.remove(CMD_FILE)
            except:
                pass
            
            if cmd:
                process_command(cmd)
                processed_count += 1
                
                # Performance logging (every 50 commands)
                if processed_count % 50 == 0:
                    logger.info(f"PERFORMANCE: {processed_count} commands processed")
        
        except Exception as e:
            logger.error(f"Command processing error: {e}")
    
    time.sleep(0.2)