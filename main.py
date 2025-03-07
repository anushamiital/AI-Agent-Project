import requests
import os
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("❌ ERROR: Hugging Face API key is missing. Please check your .env file.")

# Hugging Face Model for Response Generation
HF_MODEL = "mistralai/Mistral-7B-v0.1"  # Change model if needed

# Initialize Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def listen():
    """Capture audio and convert to text using sounddevice and speech recognition."""
    recognizer = sr.Recognizer()
    duration = 4  # Recording duration (in seconds)
    samplerate = 16000  # Sampling rate
    
    print("🎤 Listening...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    
    audio = sr.AudioData(audio_data.tobytes(), samplerate, 2)
    
    try:
        return recognizer.recognize_google(audio, language="en-IN")  # Supports Hinglish
    except sr.UnknownValueError:
        print("🔄 No speech detected. Asking user to repeat...")
        return "Mujhe samajh nahi aaya, aap phir se kahe sakte hain?"
    except sr.RequestError as e:
        print(f"⚠️ Speech recognition error: {e}")
        return f"Koi error aaya hai: {e}"

def classify_intent(text):
    """Classify user intent using Hugging Face API."""
    prompt = (
        "Classify this customer message into one of these intents:\n"
        "- demo scheduling\n- candidate interview\n- payment follow-up\n\n"
        f"Message: {text}\nIntent:"
    )

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error if request fails
        return response.json()[0]["generated_text"].strip()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Hugging Face API Error: {e}")
        return "unknown"

def generate_response(user_input, intent):
    """Generate a Hinglish response using Hugging Face API."""
    prompt = (
        f"User intent: {intent}. Generate a Hinglish response:\n"
        f"User: {user_input}\n"
        "AI:"
    )

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()[0]["generated_text"].strip()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Hugging Face API Error: {e}")
        return "Maaf kijiye, response generate karne mein dikkat ho rahi hai."

def main():
    """Main function to handle AI Cold Calls using Hugging Face API."""
    print("📞 AI Cold Call Agent Started...")

    while True:
        user_input = listen()
        print(f"🗣 User: {user_input}")

        if "exit" in user_input.lower():
            print("👋 Exiting AI Agent...")
            break

        intent = classify_intent(user_input)
        print(f"🛠 Identified Intent: {intent}")

        response = generate_response(user_input, intent)
        print(f"🤖 AI: {response}")

        speak(response)

if __name__ == "__main__":
    main()

