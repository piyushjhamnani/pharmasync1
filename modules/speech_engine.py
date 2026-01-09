import pyttsx3
import speech_recognition as sr
import threading

# Initialize TTS Engine safely
try:
    engine = pyttsx3.init()
except:
    engine = None

def speak_text_thread(text):
    """Run TTS in separate thread"""
    if not engine: return
    try:
        if engine._inLoop: engine.endLoop()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

def speak(text):
    t = threading.Thread(target=speak_text_thread, args=(text,))
    t.start()

def listen_audio():
    """
    Listens with a high tolerance for pauses.
    Returns: Text string if successful, None if silence/error.
    """
    r = sr.Recognizer()
    
    # CRITICAL FIX: Allow 3 seconds of silence before cutting off
    r.pause_threshold = 3.0 
    
    # Also adjust energy threshold for background noise
    r.dynamic_energy_threshold = True
    
    try:
        with sr.Microphone() as source:
            print("Adjusting for noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            
            print("Listening...")
            # timeout=5: Wait 5s for user to START speaking
            # phrase_time_limit=30: Allow up to 30s of total speaking time
            audio = r.listen(source, timeout=5, phrase_time_limit=30)
            
            print("Transcribing...")
            text = r.recognize_google(audio)
            return text
            
    except sr.WaitTimeoutError:
        print("Timeout: User said nothing.")
        return None # Return None to signal silence
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None