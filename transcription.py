import os
from collections import defaultdict
from pydub import AudioSegment
#import whisper
import whisperx

def format_time(seconds):
    """Convert seconds to HH-MM-SS format"""
    time_delta = timedelta(seconds=seconds)
    return str(time_delta)




'''
def transcribe_audio(diarization, audio_path, output_base_dir, transcript_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisperx.load_model("medium", device=device)
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    #whisper_model = whisper.load_model("medium")  # Adjust model size

    audio = AudioSegment.from_wav(audio_path)
    speaker_transcripts = defaultdict(list)

    for segment in diarization:
        speaker = segment["speaker_id"]
        start_time = round(segment["start"], 2)
        end_time = round(segment["end"], 2)

        # Format start and end times
        formatted_start_timie = format_time(start_time)#f"{int(start_time // 60)}-{int(start_time % 60)}"
        formatted_end_time = format_time(end_time) #f"{int(end_time // 60)}-{int(end_time % 60)}"

        speaker_dir = os.path.join(output_base_dir, f"speaker_{speaker}")
        os.makedirs(speaker_dir, exist_ok=True)

        #start_time = int(segment.start)
        #end_time = int(segment.end)
        speaker_audio = audio[start_time*1000:end_time*1000]

        segment_filename = f"{speaker}_{formatted_start_time}-{formatted_end_time}.wav".replace(":", "-") #f"{speaker}_{start_time//1000}-{end_time//1000}.wav"
        #segment_filename = segment_filename.replace(":", "-") 
        segment_path = os.path.join(speaker_dir, segment_filename)
        speaker_audio.export(segment_path, format="wav")

        # Transcribe the segment
        result = whisper_model.transcribe(segment_path)
        result_aligned = whisperx.align(result["segments"], alignment_model, metadata, segment_path, device)
        text = " ".join([seg["text"] for seg in result_aligned["segments"]])

        #text = result["text"]
        segment["text"] = text
        # Store transcript
        speaker_transcripts[speaker].append(f"{formatted_start_time} - {formatted_end_time}\n{text}\n") #f"{start_time/1000:.2f}s - {end_time/1000:.2f}s\n{text}\n")

    # Save the final transcript
    for speaker, transcript in speaker_transcripts.items():
        #transcript_path = os.path.join(output_base_dir, f"speaker_{speaker}", "transcript.txt")
        speaker_dir = os.path.join(output_base_dir, f"speaker_{speaker}")
        #print('speaker_dir',speaker_dir)
        os.makedirs(speaker_dir, exist_ok=True)
        transcript_path = os.path.join(speaker_dir, "transcript.txt")
        #print(transcript_path)
        with open(transcript_path, "w") as f:
            #for speaker, transcript in speaker_transcripts.items():
            f.write("\n".join(transcript))
'''

import whisperx
import multiprocessing
import os
import torch
from collections import defaultdict
from pydub import AudioSegment

def transcribe_segment_whisperx(segment, audio, output_base_dir, model, align_model, metadata, device):
    """Transcribes a speaker segment using WhisperX and returns the updated segment."""
    speaker = segment["speaker_id"]
    start_time = round(segment["start"], 2)
    end_time = round(segment["end"], 2)

    formatted_start_time = f"{int(start_time // 60):02d}-{int(start_time % 60):02d}"
    formatted_end_time = f"{int(end_time // 60):02d}-{int(end_time % 60):02d}"

    speaker_dir = os.path.join(output_base_dir, f"speaker_{speaker}")
    os.makedirs(speaker_dir, exist_ok=True)

    # Extract speaker audio segment
    speaker_audio = audio[start_time * 1000:end_time * 1000]
    segment_filename = f"{speaker}_{formatted_start_time}-{formatted_end_time}.wav".replace(":", "-")
    segment_path = os.path.join(speaker_dir, segment_filename)
    speaker_audio.export(segment_path, format="wav")

    # Transcribe with WhisperX
    result = model.transcribe(segment_path)
    aligned_result = whisperx.align(result["segments"], align_model, metadata, segment_path, device)

    # Extract transcribed text
    text = " ".join([seg["text"] for seg in aligned_result["segments"]])
    segment["text"] = text

    return segment

def transcribe_audio_parallel_whisperx(diarization, audio_path, output_base_dir, transcript_file):
    """Transcribes audio in parallel using WhisperX and multiprocessing."""

    # Set up WhisperX
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = whisperx.load_model("medium", device=device)
    model = whisperx.load_model("medium", device=device, compute_type="float32")
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    # Load audio
    audio = AudioSegment.from_wav(audio_path)

    # Multiprocessing pool for parallel transcription
    with multiprocessing.Pool(processes=4) as pool:  # Adjust process count as needed
        updated_segments = pool.starmap(
            transcribe_segment_whisperx,
            [(segment, audio, output_base_dir, model, align_model, metadata, device) for segment in diarization]
        )

    # Organize transcripts per speaker
    speaker_transcripts = defaultdict(list)
    for segment in updated_segments:
        speaker = segment["speaker_id"]
        formatted_start_time = f"{int(segment['start'] // 60):02d}-{int(segment['start'] % 60):02d}"
        formatted_end_time = f"{int(segment['end'] // 60):02d}-{int(segment['end'] % 60):02d}"
        text = segment["text"]

        speaker_transcripts[speaker].append(f"{formatted_start_time} - {formatted_end_time}\n{text}\n")

    # Save transcripts
    for speaker, transcript in speaker_transcripts.items():
        transcript_path = os.path.join(output_base_dir, f"speaker_{speaker}", "transcript.txt")
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        with open(transcript_path, "w") as f:
            f.write("\n".join(transcript))

