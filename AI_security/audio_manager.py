import time
import os
import pyttsx3
import speech_recognition as sr

CMD_FILE = "audio_cmd.txt"
RESP_FILE = "audio_resp.txt"

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    print("[AUDIO] TTS Engine Ready")
except Exception as e:
    print(f"[AUDIO] TTS Init Error: {e}")
    engine = None

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300  # Lower for laptop mic
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8

# Find laptop microphone
print("[AUDIO] Available microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"  {index}: {name}")

# Try to use default laptop mic (usually index 0 or 1)
LAPTOP_MIC_INDEX = None  # Set to None for default, or specific index like 1, 2, etc.

def get_microphone():
    if LAPTOP_MIC_INDEX is not None:
        return sr.Microphone(device_index=LAPTOP_MIC_INDEX)
    else:
        return sr.Microphone()  # Use system default

def speak(text):
    print(f"[AUDIO] Speaking: {text}")
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[AUDIO] TTS Error: {e}")

def listen():
    try:
        with get_microphone() as source:
            print("[AUDIO] Listening from LAPTOP microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"[AUDIO] Energy threshold: {recognizer.energy_threshold}")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            print(f"[AUDIO] Heard: {text}")
            return text
    except sr.WaitTimeoutError:
        print("[AUDIO] Timeout - no speech detected")
        return None
    except sr.UnknownValueError:
        print("[AUDIO] Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"[AUDIO] API Error: {e}")
        return None
    except Exception as e:
        print(f"[AUDIO] Listen Error: {e}")
        return None

print("[AUDIO] Audio Manager Started - Monitoring for commands...")

while True:
    if os.path.exists(CMD_FILE):
        try:
            with open(CMD_FILE, 'r') as f:
                cmd = f.read().strip()
            os.remove(CMD_FILE)
            
            print(f"[AUDIO] Command received: {cmd}")
            
            if cmd == "WARN_INTRUDER":
                speak("Warning. Intruder detected. Alert triggered.")
            
            elif cmd.startswith("WELCOME:"):
                name = cmd.split(':')[1]
                speak(f"Welcome home, {name}.")
            
            elif cmd == "ASK_NAME":
                speak("I do not recognize you. Please state your name clearly.")
                name = listen()
                if name:
                    speak(f"Thank you, {name}. You are now logged.")
                    with open(RESP_FILE, "w") as f:
                        f.write(name)
                else:
                    speak("I could not hear your name. Please try again later.")
            
            elif cmd == "FIX_BLUR":
                speak("Your face is not clear. Please step closer to the camera.")
            
            elif cmd == "FIX_DARK":
                speak("The lighting is too dark. Please turn on a light.")
            
            elif cmd == "WARN_WEAPON":
                speak("Warning. Weapon detected. Security alert.")
            
            else:
                print(f"[AUDIO] Unknown command: {cmd}")
            
        except Exception as e:
            print(f"[AUDIO] Command processing error: {e}")
    
    time.sleep(0.2)