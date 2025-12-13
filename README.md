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

Run the script with your audio file (supports `.m4a`):

```bash
python transcribe.py your_audio_file.m4a
```

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
