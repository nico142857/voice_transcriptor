import argparse
import os
import torch
import whisperx
import gc
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def check_hf_token():
    """Calculates or prompts for Hugging Face token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\n\033[93m[WARNING] Hugging Face Token not found in environment variables (HF_TOKEN).\033[0m")
        print("Speaker diarization requires a Hugging Face token and acceptance of user agreements for pyannote/speaker-diarization-3.1.")
        print("Please enter your Hugging Face Access Token (read permissions):")
        token = input("Token: ").strip()
        if not token:
            print("No token provided. Exiting.")
            exit(1)
    return token

def transcribe_audio(audio_path, hf_token, model_size="large-v2", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16"):
    """
    Transcribes and diarizes the given audio file.
    """
    print(f"Loading WhisperX model: {model_size} on {device}...")
    
    # Metal Performance Shaders (MPS) for Mac
    if torch.backends.mps.is_available():
        device = "mps"
        compute_type = "float32" # MPS often requires float32 for some ops, or at least whisperx might default to it
    elif device == "cpu":
        compute_type = "int8" # quantize for CPU speed
        
    try:
        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
        
        print(f"Transcribing {audio_path}...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        # 2. Align whisper output
        print("Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # Clean up alignment model to save memory
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. Diarization
        print("Performing speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diar_segments = diarize_model(audio)
        
        result = whisperx.assign_word_speakers(diar_segments, result)
        
        return result

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def format_output(result, output_path):
    """
    Formats the transcription result into a user-friendly text file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            speaker = segment.get("speaker", "Unknown Speaker")
            text = segment["text"].strip()
            f.write(f"[{start}] {speaker}: {text}\n")
    print(f"Transcription saved to {output_path}")

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS format."""
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Transcriptor using WhisperX")
    parser.add_argument("audio_file", help="Path to the .m4a audio file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        exit(1)

    hf_token = check_hf_token()
    
    output_file = os.path.splitext(args.audio_file)[0] + ".txt"
    
    result = transcribe_audio(args.audio_file, hf_token)
    
    if result:
        format_output(result, output_file)
