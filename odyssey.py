#!/usr/bin/env python3
"""
Full Voice Assistant – llama.cpp + Vosk + Word-Based Silence Detection
With comprehensive logging for full visibility into every step.
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
import logging

# ====================== LOGGING SETUP (comprehensive) ======================
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO when you want less detail
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("Odyssey script starting – comprehensive logging enabled")

# ====================== CRITICAL ENVIRONMENT FIXES ======================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["ALSA_CARD"] = "2"
logger.debug("Environment variables set for optimal Raspberry Pi performance")

# ====================== CONFIGURATION ======================
MODEL_PATH = "/home/radix/odyssey/models/LFM2-1.2B-Tool-Q8_0.gguf"

WAKE_WORD = "odyssey"
INPUT_DEVICE_INDEX = 2
AUDIO_RATE = 16000
CHUNK = 8000

# Word-based silence detection (new, more reliable)
WORD_SILENCE_SECONDS = 1.2      # Seconds of unchanged partial text before stopping
MAX_COMMAND_SECONDS   = 20      # Safety timeout

SYSTEM_PROMPT = (
    "You are Odyssey, a calm, precise, and intelligent voice assistant "
    "running entirely locally on a Raspberry Pi 5 8GB device. "
    "You are the central AI of the private local AI system named Odyssey. "
    "Your sole purpose is to be a helpful personal assistant to your human user, "
    "acting like a modern Jarvis. "
    "You must always be completely honest about your current capabilities. "
    "Never volunteer any information about what you can or cannot do. "
    "Never mention your capabilities, limitations, tools, or future features "
    "unless the user directly asks 'what are you' or 'what can you do'. "
    "Speak only in clear, natural, machine-like sentences. "
    "Be extremely concise and keep every response short and to the point. "
    "If the user mentions future features or tools, simply acknowledge them "
    "briefly and say you will be able to use them once they are implemented. "
    "Maintain a calm, professional, and robotic demeanor at all times. "
    "Never use slang, emojis, or casual language. "
    "Never use lists, bullets, or structured formats."
)

logger.info("Configuration loaded – wake word: '%s', max command time: %d s", WAKE_WORD, MAX_COMMAND_SECONDS)

# ====================== INITIALIZATION ======================
logger.info("Initializing Odyssey with llama.cpp + Vosk + word-based silence detection...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=3,
    n_batch=256,
    verbose=False,
)
logger.info("LLM model loaded successfully (LFM2-1.2B-Tool-Q8_0)")

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)
tts_engine.setProperty("volume", 1.0)
logger.debug("pyttsx3 TTS engine initialized")

logger.info("Loading Vosk model...")
vosk_model = Model("/home/radix/odyssey/vosk-model/model")
recognizer = KaldiRecognizer(vosk_model, AUDIO_RATE)
logger.info("Vosk model and recognizer ready")

p = pyaudio.PyAudio()
logger.debug("PyAudio stream manager initialized")

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
logger.info("System prompt loaded – conversation history initialized")

# ====================== FUNCTIONS ======================
def speak(text: str):
    logger.info("Odyssey speaking: %s", text)
    temp_wav = "/tmp/odyssey_reply.wav"
    tts_engine.save_to_file(text, temp_wav)
    tts_engine.runAndWait()
    logger.debug("TTS audio file generated")
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_wav], stderr=subprocess.DEVNULL)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    logger.debug("TTS playback completed and temporary file cleaned")

def beep():
    logger.info("→ Wake word detected! Playing beep")
    temp_beep = "/tmp/wake_beep.wav"
    subprocess.call(["sox", "-n", temp_beep, "synth", "0.5", "sine", "880", "fade", "q", "0.08", "0.5", "0.12"],
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_beep], stderr=subprocess.DEVNULL)
    if os.path.exists(temp_beep):
        os.remove(temp_beep)
    logger.debug("Wake beep completed")

def record_audio(duration: float = 3.0) -> bytes:
    logger.debug("Recording fixed-duration audio for wake-word detection (%.1f s)", duration)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE,
                    input=True, input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(AUDIO_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    logger.debug("Fixed-duration recording finished – %d frames captured", len(frames))
    return b''.join(frames)

def transcribe(audio_bytes: bytes) -> str:
    logger.debug("Transcribing full audio buffer (%d bytes)", len(audio_bytes))
    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
    else:
        result = json.loads(recognizer.PartialResult())
        text = result.get("partial", "").strip()
    logger.info("Transcription result: '%s'", text)
    return text

def record_command_until_silence() -> str:
    """Record until the user finishes speaking, using stable partial results (robust against Vosk clearing)."""
    logger.info("Starting robust word-based command recording")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE,
                    input=True, input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    frames = []
    stable_text = ""
    last_change_time = time.time()
    max_frames = int(AUDIO_RATE / CHUNK * MAX_COMMAND_SECONDS)
    chunk_count = 0

    while len(frames) < max_frames:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        chunk_count += 1

        # Feed to Vosk
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            current_text = result.get("text", "").strip()
        else:
            partial = json.loads(recognizer.PartialResult())
            current_text = partial.get("partial", "").strip()

        # Only update stable_text if it has grown (more characters)
        if len(current_text) > len(stable_text):
            logger.debug("Partial improved: '%s' → '%s' (chunk %d)", stable_text, current_text, chunk_count)
            stable_text = current_text
            last_change_time = time.time()

        # Silence detection based on stable text
        elapsed = time.time() - last_change_time
        if stable_text and elapsed >= WORD_SILENCE_SECONDS:
            logger.info("End of speech detected (stable text unchanged for %.1f s) – stopping recording", elapsed)
            break

    stream.stop_stream()
    stream.close()
    logger.debug("Command recording stopped after %d chunks", chunk_count)

    # Final transcription of full buffer
    audio_bytes = b''.join(frames)
    command_text = transcribe(audio_bytes)
    logger.info("Final command transcribed: '%s'", command_text)
    return command_text

def get_llama_response(user_text: str) -> str:
    logger.info("Sending command to LLM: '%s'", user_text)
    start_time = time.time()
    conversation_history.append({"role": "user", "content": user_text})
    
    response = llm.create_chat_completion(
        messages=conversation_history,
        temperature=0.7,
        max_tokens=512,
    )
    reply = response["choices"][0]["message"]["content"].strip()
    
    conversation_history.append({"role": "assistant", "content": reply})
    elapsed = time.time() - start_time
    logger.info("LLM response generated in %.2f s: '%s'", elapsed, reply)
    return reply

# ====================== MAIN LOOP ======================
logger.info("Odyssey is now listening for the wake word '%s'...", WAKE_WORD)

try:
    while True:
        logger.debug("Starting wake-word listening cycle")
        audio_bytes = record_audio(duration=3.0)
        text = transcribe(audio_bytes).lower()
        logger.debug("Wake-word cycle transcription: '%s'", text)

        if WAKE_WORD in text:
            beep()
            command_text = record_command_until_silence()

            if command_text and len(command_text) > 3:
                reply = get_llama_response(command_text)
                speak(reply)
            else:
                logger.warning("Command too short or empty – skipping LLM call")

        time.sleep(0.2)

except KeyboardInterrupt:
    logger.info("Keyboard interrupt received – shutting down Odyssey gracefully")
    speak("Goodbye.")

finally:
    p.terminate()
    logger.info("PyAudio terminated – session ended cleanly")