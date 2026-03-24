#!/usr/bin/env python3
"""
Full Voice Assistant – llama.cpp + Vosk (Final Stable Version)
OpenBLAS + ALSA fixes applied
"""

import os
import time
import subprocess
import json
import pyaudio
from vosk import Model, KaldiRecognizer
import pyttsx3
from llama_cpp import Llama

# ====================== CRITICAL ENVIRONMENT FIXES (MUST BE FIRST) ======================
# Prevent OpenBLAS threading conflicts and tell ALSA which mic to use
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["ALSA_CARD"] = "2"          # Your USB Mini Mic (card 2)

# ====================== CONFIGURATION ======================
MODEL_PATH = "/home/radix/odyssey/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"

WAKE_WORD = "computer"
INPUT_DEVICE_INDEX = 2
AUDIO_RATE = 16000
CHUNK = 8000

SYSTEM_PROMPT = (
    "You are Computer, a helpful, concise, and friendly voice assistant "
    "running locally on a Raspberry Pi 5. Speak naturally and briefly."
)

# ====================== INITIALIZATION ======================
print("Initializing Computer with llama.cpp + Vosk...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=3,
    n_batch=256,
    verbose=False,
)

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)
tts_engine.setProperty("volume", 1.0)

print("Loading Vosk model...")
vosk_model = Model("/home/radix/odyssey/vosk-model/model")
recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)

p = pyaudio.PyAudio()

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ====================== FUNCTIONS ======================
def speak(text: str):
    print(f"Computer: {text}")
    temp_wav = "/tmp/computer_reply.wav"
    tts_engine.save_to_file(text, temp_wav)
    tts_engine.runAndWait()
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_wav], stderr=subprocess.DEVNULL)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

def record_audio(duration: float = 3.0) -> bytes:
    """Record audio with error handling to prevent silent segfaults."""
    print("Listening...")
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=AUDIO_RATE,
                        input=True,
                        input_device_index=INPUT_DEVICE_INDEX,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"ERROR opening microphone: {e}")
        raise

    frames = []
    for _ in range(0, int(AUDIO_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    return b''.join(frames)

def transcribe(audio_bytes: bytes) -> str:
    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        return result.get("text", "").strip()
    else:
        result = json.loads(recognizer.PartialResult())
        return result.get("partial", "").strip()

def get_llama_response(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    response = llm.create_chat_completion(
        messages=conversation_history,
        temperature=0.7,
        max_tokens=512,
    )
    reply = response["choices"][0]["message"]["content"].strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

# ====================== MAIN LOOP ======================
print("\nComputer is now listening for the wake word 'computer'...")
print("Speak clearly near the USB microphone.\n")

try:
    while True:
        audio_bytes = record_audio(duration=3.0)
        text = transcribe(audio_bytes).lower()
        print(f"Transcribed: '{text}'")

        if WAKE_WORD in text:
            print("→ Wake word detected!")
            speak("Yes, I'm listening. How can I help you?")
            
            command_bytes = record_audio(duration=7.0)
            command_text = transcribe(command_bytes).strip()
            print(f"You said: {command_text}")
            
            if command_text and len(command_text) > 3:
                reply = get_llama_response(command_text)
                speak(reply)
        
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nShutting down Computer...")
    speak("Goodbye.")

finally:
    p.terminate()
    print("Session ended.")