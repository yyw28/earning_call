import os
import sys
import subprocess
import pandas as pd
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import Segment
import torch
from multiprocessing import Pool, cpu_count
import whisper
from pydub import AudioSegment


def create_directories(directories):
    """Create necessary directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def extract_company_and_quarter(filename):
    """Extract company name and quarter from filename (e.g., 'amz_q1_2023.wav' -> 'amz', 'q1_2023')."""
    parts = filename.split("_")
    if len(parts) >= 3:
        company_name = parts[0]
        quarter = "_".join(parts[1:3])  # Combine quarter and year
        return company_name, quarter

def convert_video_to_audio(video_path, output_audio_path):
    """Converts video to audio using ffmpeg."""
    if not os.path.isfile(video_path):
        print(f"Error: The specified video file '{video_path}' does not exist.")
        return None

    try:
        subprocess.run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", output_audio_path], check=True)
        print(f"Audio extracted successfully: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return None

def perform_diarization(audio_path):
    """Placeholder for speaker diarization function."""
    # Implement actual speaker diarization model
    # Load audio using pydub
    audio = AudioSegment.from_file(audio_path)

    # Trim the audio to the first 20 minutes (1200 seconds)
    duration_ms = 20 * 60 * 1000  # Convert 20 min to milliseconds
    trimmed_audio = audio[:duration_ms]

    # Save the trimmed audio as a temporary file
    trimmed_audio_path = "trimmed_audio.wav"
    trimmed_audio.export(trimmed_audio_path, format="wav")

    # Load pretrained diarization model
    pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)

    # Run diarization pipeline
    diarization = pipeline(trimmed_audio_path)
    diarization_result = [{"start": segment.start, "end": segment.end, "speaker_id": speaker} for segment, _, speaker in diarization.itertracks(yield_label=True)]
    return diarization_result

import whisper
from pydub import AudioSegment

def transcribe_audio_segment(args):
    audio_path, start_time, end_time, speaker_id = args
    # Load Whisper model (Use "base", "small", "medium", or "large" depending on accuracy vs speed tradeoff)
    model = whisper.load_model("medium")

    # Load full audio file and extract the segment
    audio = AudioSegment.from_wav(audio_path)
    segment_audio = audio[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds

    # Save the segment as a temporary file
    # Save the segment as a temporary file
    temp_audio_path = f"temp_segment_{start_time:.2f}.wav"
    segment_audio.export(temp_audio_path, format="wav")

    # Transcribe the audio segment
    result = model.transcribe(temp_audio_path)
    
    return {"timestamp": f"{start_time}-{end_time}","speaker_id": speaker_id,"text": result["text"]}


def transcribe_segments_parallel(audio_path, diarization_result):
    num_workers = min(cpu_count(), len(diarization_result))  # Limit workers to available CPU cores

    # Prepare arguments for parallel execution
    args = [(audio_path, segment["start"], segment["end"], segment["speaker_id"]) for segment in diarization_result]

    # Use multiprocessing pool for parallel execution
    with Pool(num_workers) as pool:
        transcribed_segments = pool.map(transcribe_audio_segment, args)

    return transcribed_segments


def identify_speaker_role(speaker_id):
    """Placeholder function for identifying speaker roles."""
    roles = {"S1": "CFO", "S2": "Analyst"}  # Mock roles
    return roles.get(speaker_id, "Unknown")

def extract_acoustic_features(audio_segment):
    """Placeholder for extracting acoustic features."""
    return {"pitch": 120.0, "intensity": 75.0}

def extract_related_topics(text):
    """Placeholder for extracting related topics using NLP."""
    return ["Revenue Growth", "Market Expansion"]


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


import opensmile
import librosa
import pandas as pd
import os

import subprocess
import pandas as pd

def extract_is09_features(audio_path, output_csv="is09_features.csv"):
    """
    Runs OpenSMILE to extract IS09_emotion features from an audio file.

    Args:
        audio_path (str): Path to the input audio file.
        output_csv (str): Path to save extracted features.

    Returns:
        pd.DataFrame: Extracted acoustic features.
    """
    opensmile_path = "/path/to/openSMILE-3.0.0"  # Update this path!
    config_file = f"{opensmile_path}/config/IS09_emotion.conf"

    command = [
        f"{opensmile_path}/SMILExtract",
        "-C", config_file,
        "-I", audio_path,
        "-O", output_csv
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Load extracted features from CSV
    df = pd.read_csv(output_csv, delimiter=";")
    return df




import gensim
import gensim.corpora as corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download("punkt")
nltk.download("stopwords")

def extract_topics_lda(text, num_topics=3):
    """
    Extracts topics from a given text using LDA topic modeling.

    Args:
        text (str): Input transcript text.
        num_topics (int): Number of topics to extract.

    Returns:
        list: List of extracted topics.
    """
    # Tokenization & Preprocessing
    stop_words = set(stopwords.words("english"))
    tokens = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]

    # Create Dictionary & Corpus
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    # Train LDA Model
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Extract Topics
    topics = []
    for topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
        topic_keywords = [word for word, _ in topic[1]]
        topics.append(", ".join(topic_keywords))

    return topics




def main():
    input_folder = input("Enter the path of the folder containing video files: ").strip()
    if not os.path.exists(input_folder):
        print("Error: The specified folder does not exist.")
        sys.exit(1)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.wav', '.mkv'))]
    if not video_files:
        print("No video files found in the folder.")
        sys.exit(1)

    base_output_dir = "processed_videos"
    create_directories([base_output_dir])

    for video in video_files:
        video_path = os.path.join(input_folder, video)
        company_name, quarter = extract_company_and_quarter(video)

        company_dir = os.path.join(base_output_dir, company_name,quarter)
        output_audio = os.path.join(company_dir, "output_audio.wav")
        output_txt = os.path.join(company_dir, "features.txt")

        create_directories([company_dir])
        print(f"\nProcessing: {video}")
        
        # Step 1: Convert video to audio
        audio_path = convert_video_to_audio(video_path, output_audio)
        if not audio_path:
            continue

        # Step 2: Perform Speaker Diarization
        diarization_result = perform_diarization(audio_path)
        transcribed_segments = transcribe_segments_parallel(audio_path, diarization_result)

        # Step 3: Process transcript with speaker roles, acoustic features, and related topics
        processed_transcript = []
        for segment in diarization_result:
            #timestamp = f"{segment['start']}-{segment['end']}"
            speaker_id = segment["speaker_id"]
            speaker_role = identify_speaker_role(speaker_id)
            #transcript = transcribe_audio_segment(audio_path, segment["start"], segment["end"])
            text = segment["text"]
            timestamp = segment["timestamp"]
            # Extract LDA & BERT topics
            topics_lda = extract_topics_lda(text)

            # Extract OpenSMILE IS09 Emotion features
            start_time, end_time = map(float, timestamp.split('-'))
            acoustic_features = extract_is09_features(audio_path, start_time, end_time) #extract_is09_emotion_features(audio_path, start_time, end_time)

            related_topics = extract_related_topics(text)

            processed_transcript.append({
                "timestamp": timestamp,
                "speaker_id": speaker_id,
                "speaker_role": speaker_role,
                "text": text,
                "acoustic_features": acoustic_features,
                "related_topics":topics_lda
            })

        # Step 4: Save to text file
        save_transcript_to_txt(processed_transcript, output_txt)
        print(f"Completed processing for: {video}\n")

if __name__ == "__main__":
    main()

