import argparse
import os
import torch
import whisperx
import whisperx.diarize
import gc
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file if it exists
load_dotenv()

# Enable PyTorch MPS fallback for operations not supported on Mac GPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
    
    # Authenticate with Hugging Face Hub (required for Pyannote)
    print("Authenticating with Hugging Face...")
    try:
        login(token=token)
    except Exception as e:
        print(f"Warning: Login failed: {e}")

    return token

def transcribe_audio(audio_path, hf_token, model_size="large-v2", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16", threads=4, language=None):
    """
    Transcribes and diarizes the given audio file.
    """
    # Metal Performance Shaders (MPS) for Mac
    if torch.backends.mps.is_available():
        device = "mps"
        asr_device = "cpu" # WhisperX/CTranslate2 often lacks MPS support
        compute_type = "float32" # General compute type for MPS
        asr_compute_type = "int8" # CPU ASR is faster with int8
    elif device == "cpu":
        device = "cpu"
        asr_device = "cpu"
        compute_type = "int8"
        asr_compute_type = "int8"
    else:
        asr_device = device
        asr_compute_type = compute_type

    print(f"Loading WhisperX model: {model_size} on {asr_device} (compute_type={asr_compute_type}, threads={threads})...")
    print(f"  - Alignment/Diarization device: {device}")
        
    try:
        # 1. Transcribe with original whisper (batched)
        # threads argument is passed via asr_options for CTranslate2 or directly if supported
        model = whisperx.load_model(model_size, asr_device, compute_type=asr_compute_type, threads=threads)
        
        print(f"Transcribing {audio_path}...")
        audio = whisperx.load_audio(audio_path)
        
        # Pass language if provided
        transcribe_args = {"batch_size": 16}
        if language:
            transcribe_args["language"] = language
            print(f"  - Language forced to: {language}")
            
        result = model.transcribe(audio, **transcribe_args)
        
        # 2. Align whisper output
        print("Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # Clean up alignment model to save memory
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
             torch.mps.empty_cache()

        # 3. Diarization
        print("Performing speaker diarization...")
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
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

def process_file(file_path, hf_token, args):
    """Processes a single file."""
    print(f"\n--- Processing: {file_path} ---")
    
    # Determine output path always in 'output' folder if it exists
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"{file_name}.txt")

    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}. Skipping...")
        return

    result = transcribe_audio(
        file_path, 
        hf_token, 
        model_size=args.model, 
        threads=args.threads, 
        language=args.language
    )
    
    if result:
        format_output(result, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Transcriptor using WhisperX")
    parser.add_argument("filename", help="Name of the audio file (looks in 'input/' folder first, then current path)")
    parser.add_argument("--model", default="large-v2", help="Whisper model size (small, medium, large-v2). Default: large-v2")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads for transcription. Default: 4")
    parser.add_argument("--language", help="Force language code (e.g. 'en', 'fr', 'es') to skip detection")
    
    args = parser.parse_args()
    
    # Resolve file path
    input_folder = "input"
    file_path = args.filename
    
    # 1. Check if it's in input folder (preferred)
    possible_path = os.path.join(input_folder, args.filename)
    if os.path.exists(possible_path):
        file_path = possible_path
    elif not os.path.exists(file_path):
        # 2. If not in input, and not found as absolute/relative path -> Error
        print(f"Error: File not found: {args.filename}")
        print(f"Checked: '{file_path}' and '{possible_path}'")
        exit(1)

    hf_token = check_hf_token()
    
    process_file(file_path, hf_token, args)
