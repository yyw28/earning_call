import subprocess

def convert_video_to_audio(input_video, output_audio):
    """Convert video to WAV format (Mono, 16kHz)."""
    try:
        subprocess.run(["ffmpeg", "-i", input_video, "-ac", "1", "-ar", "16000", output_audio], check=True)
        print(f"Audio extracted: {output_audio}")
    except subprocess.CalledProcessError as e:
        print(f"Error in audio conversion: {e}")

