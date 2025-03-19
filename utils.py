import os

import os
import subprocess

def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def convert_video_to_audio(video_path, output_audio_path):
    subprocess.run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", output_audio_path], check=True)

