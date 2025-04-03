api_key = "e521205915e24617a14a74bf79424798"

import assemblyai as aai
import os
import re
import csv
import heapq
import json

def process_audio_folder(folder_path, api_key):
    """
    Transcribe all audio files in a folder with speaker diarization,
    and assign per-utterance IAB topics (based on overlapping timestamps).
    Save output CSVs like COMPANY_QX_YEAR_transcript_speaker.csv.
    """
    aai.settings.api_key = api_key

    audio_files = [f for f in os.listdir(folder_path) if f.endswith((".mp3", ".wav", ".m4a"))]

    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        iab_categories=True,
        speaker_labels=True
    )

    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)

        match = re.match(r"([A-Z]+)_Q([1-4])_(\d{4})", os.path.splitext(audio_file)[0])
        if not match:
            print(f"❌ Skipping unrecognized filename format: {audio_file}")
            continue

        company, quarter, year = match.groups()
        output_filename = f"{company}_Q{quarter}_{year}_transcript_speaker.csv"
        output_path = os.path.join(folder_path, output_filename)

        print(f"🎧 Transcribing {audio_file} → {output_filename}...")

        try:
            transcript = aai.Transcriber().transcribe(audio_path, config)
        except Exception as e:
            print(f"❌ Transcription failed for {audio_file}: {e}")
            continue

        # Extract full topic list with timestamp segments
        iab_results = []
        if hasattr(transcript, "iab_categories") and transcript.iab_categories:
            #iab_results = transcript.iab_categories.get("results", [])
            iab_results = transcript.iab_categories.results if transcript.iab_categories else []


        # Save utterances to CSV
        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "speaker", "transcript", "utterance_topics"])

            for utterance in transcript.utterances:
                start = round(utterance.start / 1000, 2)
                end = round(utterance.end / 1000, 2)
                timestamp = f"{start}s - {end}s"
                speaker = f"Speaker {utterance.speaker}"
                text = utterance.text.strip()

                # Find overlapping IAB topics for this utterance
                utterance_topics = {}
                for topic in iab_results:
                    cat_start = topic.timestamp.start / 1000 #topic["timestamp"]["start"] / 1000
                    cat_end = topic.timestamp.end / 1000 #topic["timestamp"]["end"] / 1000
                    if cat_start <= end and cat_end >= start:
                        label = topic.labels[0].label #topic["labels"][0]["label"]
                        relevance = topic.labels[0].relevance #topic["labels"][0]["relevance"]
                        utterance_topics[label] = round(relevance, 4)

                # Keep top 5 topics by relevance
                top_topics = dict(heapq.nlargest(5, utterance_topics.items(), key=lambda x: x[1]))

                writer.writerow([timestamp, speaker, text, json.dumps(top_topics)])

        print(f"✅ Saved: {output_filename}\n")

process_audio_folder("./amz", api_key)

