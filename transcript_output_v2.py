import os
import re
import json
import csv
import heapq
import assemblyai as aai
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Download the sentence tokenizer
api_key = "e521205915e24617a14a74bf79424798"

def process_audio_folder(folder_path, api_key):
    """
    Transcribe all audio files in a folder with speaker diarization,
    segment utterances into sentences, and assign IAB topics.
    Save output CSVs with sentence-level rows.
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
        output_filename = f"{company.lower()}_q{quarter}_{year}_sentences.csv"
        output_path = os.path.join(folder_path, output_filename)

        print(f"🎧 Transcribing {audio_file} → {output_filename}...")

        try:
            transcript = aai.Transcriber().transcribe(audio_path, config)
        except Exception as e:
            print(f"❌ Transcription failed for {audio_file}: {e}")
            continue

        iab_results = transcript.iab_categories.results if transcript.iab_categories else []

        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "company", "quarter", "year",
                "sentence_start", "sentence_end",
                "speaker", "sentence_text", "utterance_topics"
            ])

            for utterance in transcript.utterances:
                full_text = utterance.text.strip()
                sentences = sent_tokenize(full_text)
                n = len(sentences)

                if n == 0:
                    continue

                # Divide utterance time evenly across sentences
                start = utterance.start / 1000
                end = utterance.end / 1000
                duration = (end - start) / n

                for i, sent in enumerate(sentences):
                    sent_start = round(start + i * duration, 2)
                    sent_end = round(start + (i + 1) * duration, 2)

                    # Match overlapping IAB topics
                    utterance_topics = {}
                    for topic in iab_results:
                        cat_start = topic.timestamp.start / 1000
                        cat_end = topic.timestamp.end / 1000
                        if cat_start <= sent_end and cat_end >= sent_start:
                            label = topic.labels[0].label
                            relevance = topic.labels[0].relevance
                            utterance_topics[label] = round(relevance, 4)

                    top_topics = dict(heapq.nlargest(5, utterance_topics.items(), key=lambda x: x[1]))
                    speaker = f"Speaker {utterance.speaker}"

                    writer.writerow([
                        company, quarter, year,
                        sent_start, sent_end,
                        speaker, sent, json.dumps(top_topics)
                    ])

        print(f"✅ Saved: {output_filename}\n")

process_audio_folder("./amz", api_key)

