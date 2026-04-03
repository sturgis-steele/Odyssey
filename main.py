#!/usr/bin/env python3
"""
Full Voice Assistant – llama.cpp + Vosk + Continuous Streaming Listener
Single persistent audio stream for instant wake-word + command detection.
"""

import os
import time
import subprocess
import json
import pyaudio
from vosk import Model, KaldiRecognizer
import pyttsx3
from llama_cpp import Llama
import logging

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO for normal use
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("Odyssey script starting – continuous streaming listener enabled")

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

# Word-based silence detection
WORD_SILENCE_SECONDS = 1.2
MAX_COMMAND_SECONDS = 20

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
logger.info("Initializing Odyssey with continuous streaming listener...")

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

# ====================== MAIN CONTINUOUS STREAMING LOOP ======================
logger.info("Odyssey is now listening continuously for the wake word '%s'...", WAKE_WORD)

try:
    # Open ONE persistent audio stream that remains open for the entire session
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    # Outer loop ensures the assistant never terminates after a single interaction
    while True:
        state = "wake"                    # reset to wake mode for each new cycle
        frames = []                       # only used in command mode
        stable_text = ""
        last_change_time = time.time()
        command_start_time = time.time()
        chunk_count = 0

        # Inner loop: read audio chunks continuously
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            chunk_count += 1

            # Feed chunk to Vosk
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                current_text = result.get("text", "").strip()
            else:
                partial = json.loads(recognizer.PartialResult())
                current_text = partial.get("partial", "").strip()

            lower_text = current_text.lower()

            # --------------------- WAKE MODE ---------------------
            if state == "wake":
                if WAKE_WORD in lower_text:
                    beep()
                    logger.info("Switching to command mode")
                    state = "command"
                    frames = [data]
                    stable_text = ""
                    last_change_time = time.time()
                    command_start_time = time.time()
                    if lower_text.startswith(WAKE_WORD):
                        current_text = current_text[len(WAKE_WORD):].strip()

            # --------------------- COMMAND MODE ---------------------
            elif state == "command":
                frames.append(data)

                # Update stable_text only when it grows
                if len(current_text) > len(stable_text):
                    logger.debug("Partial improved: '%s' → '%s' (chunk %d)", stable_text, current_text, chunk_count)
                    stable_text = current_text
                    last_change_time = time.time()

                # Silence detection
                elapsed = time.time() - last_change_time
                if stable_text and elapsed >= WORD_SILENCE_SECONDS:
                    logger.info("End of speech detected (stable text unchanged for %.1f s)", elapsed)
                    break

                # Safety timeout
                if time.time() - command_start_time > MAX_COMMAND_SECONDS:
                    logger.warning("Command recording reached safety timeout")
                    break

        # --------------------- PROCESS COMMAND (after silence) ---------------------
        if stable_text and len(stable_text) > 3:
            audio_bytes = b''.join(frames)
            command_text = transcribe(audio_bytes)
            command_text = command_text.replace(WAKE_WORD, "").strip()
            
            if command_text:
                reply = get_llama_response(command_text)
                speak(reply)
            else:
                logger.warning("Command too short after processing – skipping LLM")
        else:
            logger.warning("No valid command detected")

        logger.info("Returned to continuous wake-word listening")
        # Loop automatically returns to the outer while True → new wake cycle

except KeyboardInterrupt:
    logger.info("Keyboard interrupt received – shutting down Odyssey gracefully")
    speak("Goodbye.")

finally:
    try:
        stream.stop_stream()
        stream.close()
    except:
        pass
    p.terminate()
    logger.info("PyAudio terminated – session ended cleanly")