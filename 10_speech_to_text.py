
import speech_recognition as sr
from gtts import gTTS
import os

# Function to convert audio file to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Recognized Text:", text)
            print("Speech-to-Text conversion successful")
            return text
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
            return None

# Function to convert text to speech
def text_to_audio(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    print("Text-to-Speech conversion successful. Saved as", output_file)

# Main Function
def main():
    # Convert audio to text
    audio_file = r"C:\Users\Wopea\Downloads\NLP LAB\abc.wav"  # Provide the path to your audio file
    recognized_text = audio_to_text(audio_file)
    
    if recognized_text:
        # Convert text to audio
        output_audio_file = "output_audio.mp3"
        text_to_audio(recognized_text, output_audio_file)

if __name__ == "__main__":
    main()
