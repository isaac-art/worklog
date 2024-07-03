import datetime
import os
import argparse
from pydub import AudioSegment
from openai import OpenAI
import sounddevice as sd
import numpy as np
import wavio
import tempfile
import threading
import json

from settings import *

# OpenAI API setup
client = OpenAI(api_key=OPEN_AI_KEY)

# Recording setup
SAMPLE_RATE = 44100  # Sample rate in Hz


recording = True

def record_audio():
    print("Recording... Press Enter to stop.")
    audio_data = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    stream.start()
    
    def read_audio():
        while recording:
            data, overflowed = stream.read(1024)
            audio_data.append(data)
            if overflowed:
                print("Warning: Audio buffer overflowed.")
    
    thread = threading.Thread(target=read_audio)
    thread.start()
    input()  # Wait for Enter key
    global recording
    recording = False
    thread.join()
    stream.stop()
    stream.close()
    return np.concatenate(audio_data, axis=0)

def save_audio(audio_data, file_path):
    wavio.write(file_path, audio_data, SAMPLE_RATE, sampwidth=2)

def transcribe_audio(file_path):
    print("Transcribing", file_path)
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    print("Transcription complete")
    return transcript.text

def get_unique_filename(base_dir, base_name, extension):
    counter = 0
    while True:
        if counter == 0:
            file_name = f"{base_name}.{extension}"
        else:
            file_name = f"{base_name}_{chr(96 + counter)}.{extension}"
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def main():
    # Get today's date
    today = datetime.date.today()
    date_str = today.strftime('%Y_%m_%d')
    base_name = date_str
    extension = "md"
    file_name = f"{date_str}.md"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(script_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_path = get_unique_filename(log_dir, base_name, extension)
    
    sections = ["Today", "Problems", "Findings", "Questions"]
    diary_content = f"# {date_str}\n\n"
    
    for section in sections:
        print(f"Recording section: {section}")
        input("Press Enter to start recording... Press Enter again to stop.")
        global recording
        recording = True
        audio_data = record_audio()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            save_audio(audio_data, temp_audio_file.name)
            transcript = transcribe_audio(temp_audio_file.name)
        
        diary_content += f"# {section}\n\n{transcript}\n\n"
    
    with open(file_path, 'w') as file:
        file.write(diary_content)
    
    print(f"Diary file '{file_path}' created successfully.")

    # generate embeddings for the file and save to /log/embeddings/year_month_day.json
    embeddings = generate_embeddings(diary_content)
    embeddings_file_path = f"{file_path[:-3]}.json"
    with open(embeddings_file_path, 'w') as file:
        json.dump(embeddings, file)
    print(f"Embeddings file '{embeddings_file_path}' created successfully.")
    

def generate_embeddings(text):
    embeddings = client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=text
    )
    return embeddings.data[0].embedding

if __name__ == "__main__":
    main()
