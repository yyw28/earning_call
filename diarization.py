from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import Segment
import torch


'''
def perform_diarization(audio_path):
    """Perform speaker diarization on an audio file."""
    pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
    diarization = pipeline(audio_path)
    print("Speaker diarization completed.")
    return diarization
'''

def perform_diarization(audio_path):
    """Perform speaker diarization on an audio file and return structured output."""
    try:
        # Load pretrained diarization model
        pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)

        # Run diarization pipeline
        diarization_result = pipeline(audio_path)

        structured_output = []
        whisper_model = whisper.load_model("medium")  # Adjust model size
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            structured_output.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker_id": speaker,
                "text": whisper_model.transcribe(turn)# Placeholder for transcript (to be filled later)
            })

        print("Speaker diarization completed.")
        return structured_output

    except Exception as e:
        print(f"Error during diarization: {e}")
        return []


