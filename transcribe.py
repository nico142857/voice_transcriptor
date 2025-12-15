import argparse
import os
import time
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

# Check if MLX is available (Apple Silicon only)
try:
    import mlx_whisper
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

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

def transcribe_with_mlx(audio_path, model_name="mlx-community/whisper-large-v3-turbo", language=None):
    """
    Transcribe using MLX-Whisper (Apple Silicon optimized).
    Returns result dict compatible with whisperx alignment.
    """
    print(f"[MLX] Transcribing with model: {model_name}")
    
    transcribe_options = {}
    if language:
        transcribe_options["language"] = language
        print(f"  - Language forced to: {language}")
    
    start_time = time.time()
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_name,
        **transcribe_options
    )
    elapsed = time.time() - start_time
    print(f"[MLX] Transcription completed in {elapsed:.1f} seconds")
    
    return result

def transcribe_with_whisperx(audio_path, model_size, device, compute_type, threads, language):
    """
    Transcribe using WhisperX (CPU/CUDA).
    Fallback for non-Apple Silicon systems.
    """
    print(f"[WhisperX] Loading model: {model_size} on {device} (compute_type={compute_type}, threads={threads})")
    
    model = whisperx.load_model(model_size, device, compute_type=compute_type, threads=threads)
    
    print(f"[WhisperX] Transcribing {audio_path}...")
    audio = whisperx.load_audio(audio_path)
    
    transcribe_args = {"batch_size": 16}
    if language:
        transcribe_args["language"] = language
        print(f"  - Language forced to: {language}")
        
    start_time = time.time()
    result = model.transcribe(audio, **transcribe_args)
    elapsed = time.time() - start_time
    print(f"[WhisperX] Transcription completed in {elapsed:.1f} seconds")
    
    # Free up memory
    del model
    gc.collect()
    
    return result, audio

def transcribe_audio(audio_path, hf_token, model_size="large-v3-turbo", backend="auto", threads=4, language=None):
    """
    Transcribes and diarizes the given audio file.
    
    Args:
        backend: "auto" (MLX if available), "mlx", or "whisperx"
    """
    # Determine backend
    use_mlx = False
    if backend == "auto":
        use_mlx = MLX_AVAILABLE and torch.backends.mps.is_available()
    elif backend == "mlx":
        if not MLX_AVAILABLE:
            print("Error: MLX backend requested but mlx-whisper is not installed.")
            return None
        use_mlx = True
    
    # Determine device for alignment/diarization
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"\n--- Transcription Configuration ---")
    print(f"  Backend: {'MLX (GPU)' if use_mlx else 'WhisperX (CPU)'}")
    print(f"  Alignment/Diarization device: {device}")
    print(f"-----------------------------------\n")
    
    try:
        # 1. TRANSCRIPTION (the slow part - now using MLX!)
        if use_mlx:
            # Map model names for MLX
            mlx_model_map = {
                "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
                "large-v3": "mlx-community/whisper-large-v3-mlx",
                "large-v2": "mlx-community/whisper-large-v2-mlx",
                "medium": "mlx-community/whisper-medium-mlx",
                "small": "mlx-community/whisper-small-mlx",
                "base": "mlx-community/whisper-base-mlx",
                "tiny": "mlx-community/whisper-tiny-mlx",
            }
            mlx_model = mlx_model_map.get(model_size, model_size)
            result = transcribe_with_mlx(audio_path, mlx_model, language)
            # Load audio for alignment/diarization
            audio = whisperx.load_audio(audio_path)
        else:
            # Fallback to WhisperX
            asr_device = "cpu" if device == "mps" else device
            compute_type = "int8" if asr_device == "cpu" else "float16"
            result, audio = transcribe_with_whisperx(
                audio_path, model_size, asr_device, compute_type, threads, language
            )
        
        # 2. ALIGNMENT (uses MPS on Mac - fast!)
        print("Aligning transcription...")
        detected_language = result.get("language", "en")
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # Clean up alignment model
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

        # 3. DIARIZATION (uses MPS on Mac - fast!)
        print("Performing speaker diarization...")
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diar_segments = diarize_model(audio)
        
        result = whisperx.assign_word_speakers(diar_segments, result)
        
        return result

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"\n{'='*50}")
    print(f"Processing: {file_path}")
    print(f"{'='*50}")
    
    # Determine output path
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"{file_name}.txt")

    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}. Skipping...")
        return

    start_time = time.time()
    result = transcribe_audio(
        file_path, 
        hf_token, 
        model_size=args.model, 
        backend=args.backend,
        threads=args.threads, 
        language=args.language
    )
    total_time = time.time() - start_time
    
    if result:
        format_output(result, output_file)
        print(f"\n✅ Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Transcriptor - Fast transcription with speaker diarization")
    parser.add_argument("filename", help="Name of the audio file (looks in 'input/' folder first)")
    parser.add_argument("--model", default="large-v3-turbo", 
                        help="Model: tiny, base, small, medium, large-v2, large-v3, large-v3-turbo (default)")
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "whisperx"],
                        help="Backend: auto (MLX if available), mlx, whisperx")
    parser.add_argument("--threads", type=int, default=4, 
                        help="CPU threads (only for whisperx backend). Default: 4")
    parser.add_argument("--language", 
                        help="Force language code (e.g. 'en', 'fr', 'es') to skip detection")
    
    args = parser.parse_args()
    
    # Resolve file path
    input_folder = "input"
    file_path = args.filename
    
    possible_path = os.path.join(input_folder, args.filename)
    if os.path.exists(possible_path):
        file_path = possible_path
    elif not os.path.exists(file_path):
        print(f"Error: File not found: {args.filename}")
        print(f"Checked: '{file_path}' and '{possible_path}'")
        exit(1)

    hf_token = check_hf_token()
    
    process_file(file_path, hf_token, args)
