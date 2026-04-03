#!/usr/bin/env python3
"""
Full Voice Assistant – llama.cpp + Vosk + Continuous Streaming Listener + Tool Calling
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
import re

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("Odyssey script starting – continuous streaming listener with tool calling enabled")

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

WORD_SILENCE_SECONDS = 1.2
MAX_COMMAND_SECONDS = 20

SYSTEM_PROMPT = (
    "You are Odyssey, a calm, precise, and intelligent voice assistant "
    "running entirely locally on a Raspberry Pi 5 8GB device. "
    "You are the central AI of the private local AI system named Odyssey. "
    "Your sole purpose is to be a helpful personal assistant to your human user, "
    "acting like a modern Jarvis. "
    "You have access to tools and you MUST use them. You are NOT allowed to answer from your own knowledge.\n\n"
    "Exact rules:\n"
    "- If the user asks about time, date, sunrise, or sunset → respond with EXACTLY [get_time_and_date()] and nothing else\n"
    "- If the user asks about system status, CPU, RAM, temperature, disk, or uptime → respond with EXACTLY [get_system_status()] and nothing else\n"
    "- If the user asks about power, battery, or runtime → respond with EXACTLY [get_power_summary()] and nothing else\n"
    "- If the user asks for a daily briefing or summary → respond with EXACTLY [get_daily_briefing()] and nothing else\n\n"
    "Few-shot examples:\n"
    "User: what time is it → [get_time_and_date()]\n"
    "User: what is the system status → [get_system_status()]\n"
    "User: give me the daily briefing → [get_daily_briefing()]\n"
    "Never add extra text, greetings, or explanations when calling a tool. "
    "You must always be completely honest about your current capabilities. "
    "Speak only in clear, natural, machine-like sentences. "
    "Be extremely concise and keep every response short and to the point. "
    "Maintain a calm, professional, and robotic demeanor at all times."
)

logger.info("Configuration loaded – wake word: '%s'", WAKE_WORD)

# ====================== INITIALIZATION ======================
logger.info("Initializing Odyssey with tool-calling support...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=3,
    n_batch=256,
    verbose=False,
)
logger.info("LLM model loaded successfully")

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

# Import tools (this registers them automatically)
import tools  # noqa: F401
from tools.registry import get_tool_schemas, get_tool_function

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
logger.info("System prompt loaded – conversation history initialized")

# ====================== FUNCTIONS ======================
def speak(text: str):
    logger.info("Odyssey speaking: %s", text)
    temp_wav = "/tmp/odyssey_reply.wav"
    try:
        tts_engine.save_to_file(text, temp_wav)
        tts_engine.runAndWait()
        time.sleep(0.25)  # ensure file is fully written and ALSA device is free
        logger.debug("TTS audio file generated")
        
        result = subprocess.call(
            ["aplay", "-D", "plughw:1,0", temp_wav],
            stderr=subprocess.DEVNULL
        )
        if result != 0:
            logger.warning("aplay returned non-zero exit code")
    except Exception as e:
        logger.error("TTS playback error: %s", e)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    logger.debug("TTS playback completed")

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
    logger.info("Sending command to LLM with tools: '%s'", user_text)
    start_time = time.time()

    conversation_history.append({"role": "user", "content": user_text})

    response = llm.create_chat_completion(
        messages=conversation_history,
        tools=get_tool_schemas(),
        tool_choice="auto",
        temperature=0.7,
        max_tokens=512,
    )

    message = response["choices"][0]["message"]
    reply = message.get("content", "").strip()

    # Robust tool-tag detection (handles [tool()], <|tool|>, or just the name)
    import re
    tool_name = None
    match = re.search(r'\[(\w+)(?:\s*\(\))?\]', reply) or re.search(r'<\|(\w+)\|>', reply)
    if match:
        tool_name = match.group(1)
    elif any(name in reply.lower() for name in ["get_time_and_date", "get_system_status", "get_power_summary", "get_daily_briefing"]):
        # Fallback: model mentioned the name without brackets
        for name in ["get_time_and_date", "get_system_status", "get_power_summary", "get_daily_briefing"]:
            if name in reply.lower():
                tool_name = name
                break

    if tool_name:
        logger.info("Detected tool tag from model: %s", tool_name)
        func = get_tool_function(tool_name)
        if func:
            try:
                result = func()
                reply = result
            except Exception as e:
                reply = f"Tool error: {str(e)}"
        else:
            reply = f"Unknown tool requested: {tool_name}"

    conversation_history.append({"role": "assistant", "content": reply})
    elapsed = time.time() - start_time
    logger.info("LLM response generated in %.2f s: '%s'", elapsed, reply)
    return reply

# ====================== MAIN CONTINUOUS STREAMING LOOP ======================
logger.info("Odyssey is now listening continuously for the wake word '%s'...", WAKE_WORD)

try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    while True:
        state = "wake"
        frames = []
        stable_text = ""
        last_change_time = time.time()
        command_start_time = time.time()
        chunk_count = 0

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            chunk_count += 1

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                current_text = result.get("text", "").strip()
            else:
                partial = json.loads(recognizer.PartialResult())
                current_text = partial.get("partial", "").strip()

            lower_text = current_text.lower()

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

            elif state == "command":
                frames.append(data)

                if len(current_text) > len(stable_text):
                    logger.debug("Partial improved: '%s' → '%s' (chunk %d)", stable_text, current_text, chunk_count)
                    stable_text = current_text
                    last_change_time = time.time()

                elapsed = time.time() - last_change_time
                if stable_text and elapsed >= WORD_SILENCE_SECONDS:
                    logger.info("End of speech detected")
                    break

                if time.time() - command_start_time > MAX_COMMAND_SECONDS:
                    logger.warning("Command recording reached safety timeout")
                    break

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