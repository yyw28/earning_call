import os
import sys
from auth import authenticate_huggingface
from diarization import perform_diarization
from transcription import transcribe_audio_parallel_whisperx,transcribe_segment_whisperx #transcribe_audio
from utils import convert_video_to_audio, create_directories
import subprocess
import pandas as pd

def create_directories(directories):
    """Create necessary directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def extract_company_code(filename):
    """Extract the company code from the filename (first part before space or underscore)."""
    return filename.split()[0] if " " in filename else filename.split("_")[0]

def convert_video_to_audio(video_path, output_audio_path):
    """Converts video to audio using ffmpeg."""
    if not os.path.isfile(video_path):
        print(f"Error: The specified video file '{video_path}' does not exist.")
        return

    try:
        subprocess.run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", output_audio_path], check=True)
        print(f"Audio extracted successfully: {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")


def identify_speaker_role(speaker_id):
    """Placeholder function for identifying speaker roles."""
    roles = {"S1": "CFO", "S2": "Analyst"}  # Mock roles
    return roles.get(speaker_id, "Unknown")

def save_transcript_to_txt(transcript_data, output_path):
    """Save processed transcript data into a structured text file."""
    with open(output_path, "w") as file:
        file.write("Timestamp\tSpeaker ID\tSpeaker Role\tTranscript\tAcoustic Features\tRelated Topics\n")
        for row in transcript_data:
            acoustic_features = row.get("acoustic_features", {})
            related_topics = row.get("related_topics", [])
            acoustic_str = f"Pitch: {acoustic_features.get('pitch', 'N/A')}, Intensity: {acoustic_features.get('intensity', 'N/A')}"
            topics_str = ", ".join(related_topics)
            file.write(f"{row['timestamp']}\t{row['speaker_id']}\t{row['speaker_role']}\t{row['text']}\t{acoustic_str}\t{topics_str}\n")
    print(f"Transcript saved to {output_path}")


def main():
    # Get video input folder from user
    input_folder = input("Enter the path of the folder containing video files: ").strip()

    if not os.path.exists(input_folder):
        print("Error: The specified folder does not exist.")
        sys.exit(1)

    # Get all video files in the folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.wav', '.mkv'))]

    if not video_files:
        print("No video files found in the folder.")
        sys.exit(1)

    # Base directory for processed videos
    base_output_dir = "processed_videos"
    create_directories([base_output_dir])

    for video in video_files:
        video_path = os.path.join(input_folder, video)
        company_code = extract_company_code(video)

        # Create company-specific directories
        company_dir = os.path.join(base_output_dir, company_code)
        output_audio = os.path.join(company_dir, "output_audio.wav") 
        output_txt = os.path.join(company_dir, "features.txt")
        output_dir = os.path.join(company_dir, "segmented_audios")
        transcript_file = os.path.join(company_dir, "transcript.txt")
        output_base_dir = os.path.join(company_dir, "speakers")

        create_directories([company_dir, output_dir, output_base_dir])
        print(f"\nProcessing: {video}")
        # Step 1: Authenticate Hugging Face
        authenticate_huggingface()
        # Step 2: Convert video to audio
        convert_video_to_audio(video_path, output_audio)
        # Step 3: Perform Speaker Diarization
        diarization_result = perform_diarization(output_audio)

        # Step 4: Transcribe Segments
        #transcribe_audio(diarization_result, output_audio, output_base_dir, transcript_file)
        transcribe_audio_parallel_whisperx(diarization_result, output_audio, output_base_dir, transcript_file)
        print(f"Completed processing for: {video}\n")

        # Step 3: Process transcript with speaker roles, acoustic features, and related topics
        processed_transcript = []
        for segment in diarization_result:
            timestamp = f"{segment['start']}-{segment['end']}"
            speaker_id = segment["speaker_id"]
            speaker_role = identify_speaker_role(speaker_id)
            text = segment["text"]
            acoustic_features = extract_acoustic_features(audio_path)
            related_topics = extract_related_topics(text)

            processed_transcript.append({
                "timestamp": timestamp,
                "speaker_id": speaker_id,
                "speaker_role": speaker_role,
                "text": text,
                "acoustic_features": acoustic_features,
                "related_topics": related_topics
            })

        # Step 4: Save to text file
        save_transcript_to_txt(processed_transcript, output_txt)
        print(f"Completed processing for: {video}\n")        

if __name__ == "__main__":
    main()

