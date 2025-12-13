# Voice Transcriptor

A Python tool to transcribe and diarize (distinguish speakers) audio files in English and French using OpenAI's Whisper (via WhisperX) and Pyannote Audio.

## Features
-   **Transcription**: High-accuracy transcription using Whisper large-v2 models.
-   **Speaker Diarization**: Identifies and labels different speakers (Speaker 1, Speaker 2, etc.).
-   **Word-level Timestamps**: Precise alignment of text and audio.
-   **Output**: Generates user-friendly text files.

## Prerequisites
-   **Python 3.10**
-   **Micromamba** or Conda
-   **Hugging Face Account**: Required for the speaker diarization model.
    1.  Create an account at [huggingface.co](https://huggingface.co).
    2.  Create an [Access Token](https://huggingface.co/settings/tokens) (Read role).
    3.  Accept the user agreement for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd voice_transcriptor
    ```

2.  **Set up the environment**:
    If you already have a `transcripcion` environment, update it:
    ```bash
    micromamba env update -n transcripcion --file environment.yml --prune
    ```
    Or create a new one:
    ```bash
    micromamba create -f environment.yml
    ```

3.  **Activate the environment**:
    ```bash
    micromamba activate transcripcion
    ```

4.  **Configuration**:
    Create a `.env` file in the project root and add your Hugging Face token:
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```env
    HF_TOKEN=hf_your_token_here
    ```

## Usage

## Usage

1.  Put your audio files (e.g., `interview.m4a`) into the `input` folder.
2.  Run the script with the filename:
    ```bash
    python transcribe.py interview.m4a
    ```
    *(Note: You don't need to type `input/interview.m4a`, just the name)*
3.  Collect your transcriptions from the `output` folder.

### Output
The transcription will be saved as a `.txt` file in the same directory as the input audio.

**Example Output (`your_audio_file.txt`):**
```text
[00:00:05] Speaker 01: Bonjour, ceci est un test.
[00:00:10] Speaker 02: And this is the response in English.
```

## Troubleshooting
-   **First Run Slowness**: The first time you run the script, it downloads the Whisper and Pyannote models (several GBs). Subsequent runs will be faster.
-   **Memory Issues**: If you run out of memory, try editing `transcribe.py` to use a smaller model (e.g., `model_size="medium"` or `model_size="small"`).

## Dependency Notes
**Critical**: This project relies on a specific combination of libraries to work on macOS (Apple Silicon) without errors.
-   **PyTorch/Torchaudio 2.5.1**: Used instead of the newer 2.8+ versions often requested by pip, to ensure stability with `micromamba` and MPS (Metal Performance Shaders).
-   **Transformers 4.48.0**: Pinned to avoid import errors with PyTorch 2.5.1.
-   **WhisperX**: Installed from source to get the latest features.

**Warning**: Do not manually `pip install --upgrade` these packages, or you may break the environment (e.g., getting `symbol not found` or `OSError: libjpeg` errors). Always use `micromamba env update --file environment.yml`.
