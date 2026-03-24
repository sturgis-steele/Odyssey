#!/usr/bin/env python3
"""
Full Voice Assistant – llama.cpp + Vosk + Simple RMS Silence Detection
No extra dependencies (webrtcvad removed)
"""

import os
import time
import subprocess
import json
import math
import pyaudio
from vosk import Model, KaldiRecognizer
import pyttsx3
from llama_cpp import Llama

# ====================== CRITICAL ENVIRONMENT FIXES ======================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["ALSA_CARD"] = "2"

# ====================== CONFIGURATION ======================
MODEL_PATH = "/home/radix/odyssey/models/LFM2-1.2B-Tool-Q8_0.gguf"

WAKE_WORD = "odyssey"
INPUT_DEVICE_INDEX = 2
AUDIO_RATE = 16000
CHUNK = 8000

# Silence detection settings (tune these if needed)
ENERGY_THRESHOLD = 300      # Lower = more sensitive to quiet speech (try 200–500)
SILENCE_SECONDS = 1.2       # How long of silence before we stop recording
MAX_COMMAND_SECONDS = 60    # Safety timeout

SYSTEM_PROMPT = (
    "You are Odyssey, a calm, precise, and intelligent voice assistant "
    "running entirely locally on a Raspberry Pi 5 8GB device. "
    "You are the central AI of the private local AI system named Odyssey. "
    "Your sole purpose is to be a helpful personal assistant to your human user, "
    "acting like a modern Jarvis. "
    "You must always be completely honest about your current capabilities. "
    "Never claim you can perform an action, set something up, or use a tool "
    "that you do not currently have. "
    "Never volunteer information about your capabilities, limitations, or functions. "
    "Never explain what you can or cannot do unless the user directly asks "
    "'what are you' or 'what can you do'. "
    "Be extremely concise. Keep every response short and to the point. "
    "If the user mentions future features or tools, you may acknowledge them "
    "and say you will be able to use them once they are implemented. "
    "Speak in clear, natural, machine-like sentences. "
    "Be concise, slightly witty when appropriate, but always maintain a calm "
    "and professional robotic demeanor. "
    "Never use slang, emojis, or overly casual language. "
    "Keep most responses relatively short and engaging. "
    "Never give long structured lists, step-by-step instructions, tutorials, "
    "or code examples unless the user explicitly asks for them."
)

# ====================== INITIALIZATION ======================
print("Initializing Odyssey with llama.cpp + Vosk + Simple RMS Silence Detection...")

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
    print(f"Odyssey: {text}")
    temp_wav = "/tmp/odyssey_reply.wav"
    tts_engine.save_to_file(text, temp_wav)
    tts_engine.runAndWait()
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_wav], stderr=subprocess.DEVNULL)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

def beep():
    print("→ Wake word detected! (beep)")
    temp_beep = "/tmp/wake_beep.wav"
    subprocess.call(["sox", "-n", temp_beep, "synth", "0.5", "sine", "880", "fade", "q", "0.08", "0.5", "0.12"],
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_beep], stderr=subprocess.DEVNULL)
    if os.path.exists(temp_beep):
        os.remove(temp_beep)

def record_audio(duration: float = 3.0) -> bytes:
    """Short fixed recording used only for wake-word detection."""
    print("Listening...")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE,
                    input=True, input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(AUDIO_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    return b''.join(frames)

def transcribe(audio_bytes: bytes) -> str:
    """Convert audio bytes to text using Vosk."""
    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        return result.get("text", "").strip()
    else:
        result = json.loads(recognizer.PartialResult())
        return result.get("partial", "").strip()

def get_rms(audio_chunk: bytes) -> float:
    """Calculate Root Mean Square energy of an audio chunk."""
    rms = 0.0
    count = len(audio_chunk) // 2
    for i in range(count):
        sample = int.from_bytes(audio_chunk[i*2:i*2+2], "little", signed=True)
        rms += sample * sample
    return math.sqrt(rms / count) if count > 0 else 0.0

def record_command_until_silence() -> str:
    """Record until the user stops speaking (simple energy-based silence detection)."""
    print("Listening for your command... (speak naturally — I will stop when you finish)")

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE,
                    input=True, input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    frames = []
    silence_frames = 0
    max_frames = int(AUDIO_RATE / CHUNK * MAX_COMMAND_SECONDS)
    silence_threshold_frames = int(AUDIO_RATE / CHUNK * SILENCE_SECONDS)

    while len(frames) < max_frames:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        rms = get_rms(data)
        if rms < ENERGY_THRESHOLD:
            silence_frames += 1
        else:
            silence_frames = 0

        if silence_frames >= silence_threshold_frames:
            print("Silence detected — stopping recording")
            break

    stream.stop_stream()
    stream.close()

    audio_bytes = b''.join(frames)
    command_text = transcribe(audio_bytes)
    print(f"You said: {command_text}")
    return command_text

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
print("\nOdyssey is now listening for the wake word 'odyssey'...")
print("Speak clearly near the USB microphone.\n")

try:
    while True:
        audio_bytes = record_audio(duration=3.0)
        text = transcribe(audio_bytes).lower()
        print(f"Transcribed: '{text}'")

        if WAKE_WORD in text:
            beep()

            # Natural listening — stops when you pause
            command_text = record_command_until_silence()

            if command_text and len(command_text) > 3:
                reply = get_llama_response(command_text)
                speak(reply)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nShutting down Odyssey...")
    speak("Goodbye.")

finally:
    p.terminate()
    print("Session ended.")