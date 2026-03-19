#!/usr/bin/env python3
"""
Computer - Vosk Version (Low CPU + High Accuracy)
Tested for your USB Mini Mic (card 2) + LCD speakers (plughw:1,0)
"""

import os
import time
import ollama
import numpy as np
import pyaudio
import subprocess
from vosk import Model, KaldiRecognizer
import json
import pyttsx3

# Use your USB mic
os.environ['ALSA_CARD'] = '2'

WAKE_WORD = "computer"
LLM_MODEL = "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest"
INPUT_DEVICE_INDEX = 2
AUDIO_RATE = 16000          # Vosk models are 16 kHz
CHUNK = 8000

SYSTEM_PROMPT = (
    "You are Computer, a helpful, concise, and friendly voice assistant "
    "running locally on a Raspberry Pi 5. Speak naturally and briefly."
)

print("Initializing Computer with Vosk...")

# Load Vosk model
model = Model("/home/radix/odyssey/vosk-model/model")
recognizer = KaldiRecognizer(model, AUDIO_RATE)

# TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)
tts_engine.setProperty("volume", 1.0)

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

p = pyaudio.PyAudio()

def speak(text: str):
    print(f"Computer: {text}")
    temp_wav = "/tmp/computer_reply.wav"
    tts_engine.save_to_file(text, temp_wav)
    tts_engine.runAndWait()
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_wav], stderr=subprocess.DEVNULL)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

def record_audio(duration: float = 3.0) -> bytes:
    print("Listening...")
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)
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

def get_ollama_response(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    response = ollama.chat(model=LLM_MODEL, messages=conversation_history, options={"temperature": 0.7})
    reply = response["message"]["content"].strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

print("Computer is ready and listening for 'computer'...")
print("Speak clearly near the microphone.")

try:
    while True:
        audio_bytes = record_audio(duration=3.0)
        text = transcribe(audio_bytes).lower()
        print(f"Transcribed: '{text}'")   # Debug line

        if "computer" in text:
            print("→ Wake word detected!")
            speak("Yes, I'm listening. How can I help you?")
            
            command_bytes = record_audio(duration=7.0)
            command_text = transcribe(command_bytes).strip()
            print(f"You said: {command_text}")
            
            if command_text and len(command_text) > 3:
                reply = get_ollama_response(command_text)
                speak(reply)
        
        time.sleep(0.2)   # Low CPU now

except KeyboardInterrupt:
    print("\nComputer shutting down.")
    speak("Goodbye.")
finally:
    p.terminate()
