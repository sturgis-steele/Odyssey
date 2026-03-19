#!/usr/bin/env python3
"""
Odyssey - Local Voice Assistant (Stable version for Pi 5 + your USB Mic)
"""

import os
import time
import ollama
import numpy as np
import pyaudio
import subprocess
from faster_whisper import WhisperModel
import pyttsx3

# Suppress ALSA warnings (your mic + speakers)
os.environ['ALSA_CARD'] = '2'

WAKE_WORD = "hey odyssey"
LLM_MODEL = "lfm2.5:1.2b-instruct"
INPUT_DEVICE_INDEX = 2          # Your USB microphone
AUDIO_RATE = 44100
CHUNK = 1024

SYSTEM_PROMPT = (
    "You are Odyssey, a helpful, concise, and friendly voice assistant "
    "running locally on a Raspberry Pi 5. "
    "Speak naturally and briefly."
)

print("Initializing Odyssey...")

whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)
tts_engine.setProperty("volume", 1.0)

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

p = pyaudio.PyAudio()

def speak(text: str):
    print(f"Odyssey: {text}")
    temp_wav = "/tmp/odyssey_reply.wav"
    tts_engine.save_to_file(text, temp_wav)
    tts_engine.runAndWait()
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_wav], stderr=subprocess.DEVNULL)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

def record_audio(duration: float = 3.0) -> np.ndarray:
    print("Listening...")
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=AUDIO_RATE,
                        input=True,
                        input_device_index=INPUT_DEVICE_INDEX,
                        frames_per_buffer=CHUNK)
        
        frames = []
        for _ in range(0, int(AUDIO_RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        stream.stop_stream()
        stream.close()
        
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        return audio
    except Exception as e:
        print(f"Recording error: {e}")
        return np.zeros(16000, dtype=np.float32)

def transcribe(audio_data: np.ndarray) -> str:
    try:
        segments, _ = whisper_model.transcribe(audio_data, beam_size=5, language="en")
        return " ".join(segment.text for segment in segments).strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def get_ollama_response(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    response = ollama.chat(model=LLM_MODEL, messages=conversation_history, options={"temperature": 0.7})
    reply = response["message"]["content"].strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

print(f"Odyssey is ready and listening for '{WAKE_WORD}'...")
print("Speak clearly near the microphone.")

try:
    while True:
        audio = record_audio(duration=3.0)
        text = transcribe(audio).lower()
        
        if WAKE_WORD in text or "odyssey" in text:
            print("→ Wake word detected!")
            speak("Yes, how can I help you?")
            
            command_audio = record_audio(duration=8.0)
            command_text = transcribe(command_audio)
            
            if command_text and len(command_text) > 3:
                print(f"You said: {command_text}")
                reply = get_ollama_response(command_text)
                speak(reply)
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nOdyssey shutting down.")
    speak("Goodbye.")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    p.terminate()
