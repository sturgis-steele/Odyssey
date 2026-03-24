#!/usr/bin/env python3
"""
Bare-Bones Llama.cpp Voice Assistant Test Script
Fully commented for learning and maintenance.
Project location: ~/odyssey/llama-ccp/computer_llama.py
"""

import os                  # Used to work with files and folders (e.g., deleting temp audio file)
import time                # Used for small delays to reduce CPU usage
import subprocess          # Used to call the "aplay" command to play sound through your speakers
import pyttsx3             # Text-to-Speech engine (converts text into spoken audio)
from llama_cpp import Llama  # The main library that gives us access to llama.cpp (fast local LLM inference)


# ====================== CONFIGURATION ======================
# This section contains settings you can easily change

# Path to the GGUF model file we downloaded
# Make sure this path exactly matches your filename in ~/odyssey/models/
MODEL_PATH = "/home/radix/odyssey/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"

# System prompt tells the model how to behave
SYSTEM_PROMPT = (
    "You are Computer, a helpful, concise, and friendly voice assistant "
    "running locally on a Raspberry Pi 5. Speak naturally and briefly."
)

# Wake word (kept for future voice integration)
WAKE_WORD = "computer"


# ====================== INITIALIZATION ======================
# Everything in this section runs once when the script starts

print("Initializing Computer with llama.cpp...")

# Load the LLM model using llama.cpp
# This is the most important part — it loads the quantized model into RAM
llm = Llama(
    model_path=MODEL_PATH,      # Path to the .gguf file
    n_ctx=4096,                 # Maximum context length (how much conversation history the model can remember)
    n_threads=4,                # Number of CPU threads — set to 4 because Raspberry Pi 5 has 4 performance cores
    n_batch=512,                # How many tokens to process at once (good balance for speed on Pi)
    verbose=False,              # Set to True if you want detailed debug information during inference
)

# Initialize the Text-to-Speech engine (pyttsx3)
tts_engine = pyttsx3.init()

# Configure voice settings
tts_engine.setProperty("rate", 170)    # Speaking speed (170 words per minute is natural)
tts_engine.setProperty("volume", 1.0)  # Full volume

# Start conversation history with the system prompt
# This tells the model who it is from the very beginning
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ====================== FUNCTIONS ======================
# Reusable functions that perform specific tasks

def speak(text: str):
    """
    Converts text to speech and plays it through your USB speakers (plughw:1,0).
    This is the same speak() function from your original script.
    """
    print(f"Computer: {text}")                     # Print the response to the terminal

    temp_wav = "/tmp/computer_reply.wav"           # Temporary file to store generated audio

    # Generate speech and save it as a WAV file
    tts_engine.save_to_file(text, temp_wav)
    tts_engine.runAndWait()                        # Actually generate the audio

    # Play the audio file using aplay (your speaker setup)
    subprocess.call(["aplay", "-D", "plughw:1,0", temp_wav], 
                    stderr=subprocess.DEVNULL)     # Hide any error messages

    # Clean up — delete the temporary WAV file
    if os.path.exists(temp_wav):
        os.remove(temp_wav)


def get_llama_response(user_text: str) -> str:
    """
    Sends the user's message to the LLM and returns the assistant's reply.
    Uses the same chat format as your original Ollama code.
    """
    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Call the model to generate a response
    response = llm.create_chat_completion(
        messages=conversation_history,   # Send full conversation so far
        temperature=0.7,                 # Controls creativity (0.7 is balanced)
        max_tokens=512,                  # Maximum length of the reply
    )
    
    # Extract the text reply from the model's response
    reply = response["choices"][0]["message"]["content"].strip()
    
    # Add the assistant's reply to the conversation history
    conversation_history.append({"role": "assistant", "content": reply})
    
    return reply


# ====================== MAIN LOOP ======================
# This is where the program runs continuously

print("\nComputer is ready for testing.")
print("Type your message and press Enter. Type 'exit' to quit.\n")

try:
    while True:                                # Keep running until user exits
        # Get input from the keyboard
        user_input = input("You: ").strip()
        
        # Allow graceful exit
        if user_input.lower() == "exit":
            break
        
        # Skip empty messages
        if not user_input:
            continue

        # Optional: detect wake word (useful when we add voice later)
        if WAKE_WORD in user_input.lower():
            print("→ Wake word detected!")
            speak("Yes, I'm listening. How can I help you?")
            user_input = input("You (command): ").strip()

        # Only process meaningful input
        if user_input and len(user_input) > 3:
            reply = get_llama_response(user_input)   # Get AI response
            speak(reply)                             # Speak the response

        time.sleep(0.2)      # Small delay to keep CPU usage low

except KeyboardInterrupt:
    # This runs if you press Ctrl+C
    print("\nShutting down...")
    speak("Goodbye.")

finally:
    print("Session ended.")
