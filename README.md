# Voice Transcriptor

A Python-based tool for automated speech-to-text transcription and speaker diarization using [WhisperX](https://github.com/m-bain/whisperX).

This project relies on `whisperx` for state-of-the-art ASR (Automatic Speech Recognition) and `pyannote-audio` for speaker diarization (distinguishing between unique speakers). It is optimized for execution on macOS (Apple Silicon) using MPS (Metal Performance Shaders).

## Prerequisites

Before proceeding, ensure the following are available on your system:

-   **Python 3.10**: The codebase relies on libraries compatible with Python 3.10.
-   **Micromamba**: A fast package manager (compatible with Conda) is required for environment isolation.
-   **Hugging Face Account**: Required to download the gated diarization models.

### Hugging Face Token Configuration
The speaker diarization pipeline relies on the `pyannote/speaker-diarization-3.1` model, which is a gated repository.

1.  **Create an Account**: Register at [huggingface.co](https://huggingface.co).
2.  **Generate an Access Token**:
    -   Navigate to [Settings > Tokens](https://huggingface.co/settings/tokens).
    -   Create a new **Fine-grained** token.
    -   Under **Repositories permissions**, ensure you check the box for: **"Read access to contents of all public gated repos"**.
3.  **Accept Model Agreements**:
    -   Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the license.
    -   Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the license.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd voice_transcriptor
    ```

2.  **Initialize the Environment**:
    Create the `transcripcion` environment using the provided lock file to ensure dependency compatibility.
    ```bash
    micromamba create -f environment.yml
    ```
    *Note: This installs specific pinned versions of PyTorch (2.5.1), Torchaudio (2.5.1), and Transformers (4.48.0) to ensure stability on macOS.*

3.  **Activate the Environment**:
    ```bash
    micromamba activate transcripcion
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the project root to store your credentials secure.
    ```bash
    echo "HF_TOKEN=hf_your_token_here" > .env
    ```

## Usage

The project utilizes a strict input/output directory structure for workflow organization.

1.  **Prepare Input**: Place your audio files (e.g., `.m4a`, `.mp3`) in the `input/` directory.
2.  **Execute Transcription**:
    Run the script providing the filename as an argument.
    ```bash
    python transcribe.py filename.m4a
    ```
    *(The script will automatically resolve the path to `input/filename.m4a`)*

3.  **Retrieve Output**: The transcribed text file will be generated in the `output/` directory (e.g., `output/filename.txt`).

### Output Format
The output file follows a standard timestamped format:
```text
[00:00:05] Speaker 01: Bonjour, ceci est un test.
[00:00:10] Speaker 02: And this is the response in English.
```

## Advanced Usage (Performance)

To optimize transcription speed, especially on CPU, you can use additional arguments:

```bash
python transcribe.py filename.m4a --model medium --threads 8 --language fr
```

| Argument | Description | Default | Recommended for Speed |
| :--- | :--- | :--- | :--- |
| `--model` | Whisper model size (`small`, `medium`, `large-v2`). `medium` is ~3x faster than `large-v2` with good accuracy. | `large-v2` | `medium` |
| `--threads` | Number of CPU threads to use. Set this to your number of performance cores (e.g., 8 on M1/M2/M3 Pro/Max). | `4` | `8` or `10` |
| `--language` | Force language code (e.g., `en`, `fr`, `es`) to skip the 30-second detection phase. | Auto-detect | Explicit language |

## Technical Dependency Notes

This project requires a precise combination of library versions to function correctly on macOS/MPS.

-   **PyTorch**: Pinned to `2.5.1`. Newer versions (2.8+) introduced compatibility issues with `whisperx` dependencies on this architecture.
-   **Transformers**: Pinned to `4.48.0` to maintain compatibility with PyTorch 2.5.1.
-   **Device Handling**: The script implements a hybrid execution model:
    -   **ASR (Transcription)**: Forces execution on **CPU** (int8) to bypass CTranslate2 MPS limitations.
    -   **Diarization**: Executed on **MPS (GPU)** for performance.
    -   **Fallback**: `PYTORCH_ENABLE_MPS_FALLBACK=1` is automatically set to handle unsupported operations seamlessly.

**Caution**: Do not manually `pip install --upgrade` the dependencies. Rely on `micromamba env update` with the provided `environment.yml`.
