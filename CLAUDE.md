# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Finnish radiological TTS training data pipeline. Takes transcribed radiological dictations (CSV) and produces WAV audio files using XTTS v2 voice cloning from Mozilla Common Voice Finnish speakers. Output is used to train Finnish speech/language models on radiological vocabulary.

## Environment

- **OS**: Windows 11, shell: bash (WSL or Git Bash)
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) — always target CUDA, never CPU-only
- **Python**: 3.11+, scripts run from repo root

## Setup

Coqui TTS vaatii Python <3.12, joten käytetään conda-ympäristöä:

```bash
conda create -n tts-radio python=3.11 -y
conda activate tts-radio

# PyTorch must be installed with CUDA before other packages
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# HuggingFace login required for Common Voice download
huggingface-cli login
# or: set HF_TOKEN=hf_xxxxx
```

Skriptit ajetaan aina `tts-radio`-ympäristössä:
```bash
conda activate tts-radio
python 1_prepare_voices.py
```

## Cloud GPU (RunPod)

Kun ajetaan RunPodissa (PyTorch-template, RTX 4090, ≥50 GB levy):

```bash
# 1. Kloonaa repo JupyterLabin terminaalissa
git clone https://github.com/<käyttäjä>/saneludata .

# 2. Asenna riippuvuudet (PyTorch + CUDA jo valmiina templatessa)
bash cloud_setup.sh

# 3. Kirjaudu HuggingFaceen
huggingface-cli login

# 4. Aja pipeline (sama kuin paikallisesti)
python 1_prepare_voices.py
python 2_generate_audio.py data/transcriptions/sanelut.csv

# 5. Siirrä data HuggingFace Hubiin
python 3_push_to_hub.py --repo käyttäjänimi/radiologia-audio
```

CSV-tiedosto: lataa JupyterLabin Upload-napilla tai kopioi tekstinä.

## Pipeline Commands

```bash
# Step 1 — run once: build reference voice files from local Common Voice download
# Download Finnish dataset first: https://commonvoice.mozilla.org/fi/datasets
# Extract tar.gz, then:
python 1_prepare_voices.py --cv-path /path/to/cv-corpus-XX.0/fi
python 1_prepare_voices.py --cv-path /path/to/cv-corpus-XX.0/fi --speakers 30 --min-clips 5

# Step 2 — generate audio from transcriptions CSV
python 2_generate_audio.py data/transcriptions/sanelut.csv
python 2_generate_audio.py data/transcriptions/sanelut.csv --speakers 10 --seed 123
```

Step 2 is resumable: already-generated WAV files are skipped automatically.

## Data Layout

```
data/
  transcriptions/   ← input CSV files (id, text columns)
  reference_voices/ ← speaker_000.wav … speaker_N.wav + manifest.json
  output/           ← generated WAVs named {id}_{speaker_id}.wav + metadata.csv
```

**Input CSV format** (`id`, `text` columns required, UTF-8):
```
id,text
lausunto_001,"Keuhkojen posteroanteriorinen röntgenkuva..."
```

**Output metadata.csv** columns: `id, audio_file, speaker_id, duration_sec, text`

## Architecture

Two independent scripts connected only by `data/reference_voices/manifest.json`:

**`1_prepare_voices.py`**
Reads Mozilla Common Voice Finnish from a locally downloaded tar.gz (commonvoice.mozilla.org). As of October 2025, Common Voice is no longer available on HuggingFace. Reads `validated.tsv` and `clips/*.mp3`, groups by `client_id`, filters by minimum clip count, selects N speakers with gender diversity, concatenates clips into per-speaker reference WAV files at 24000 Hz. Also concatenates `sentence` fields into `ref_text` stored in `manifest.json` — required by F5-TTS.

**`2_generate_audio.py`**
Loads F5-TTS via `f5_tts.api.F5TTS`. For each CSV row, picks a random speaker from the manifest (which now includes `ref_text` — the transcript of the reference audio), calls `split_sentences()` to break the text into sentence chunks, synthesizes each sentence separately with `tts.infer(ref_file, ref_text, gen_text)`, then concatenates with 450 ms pauses. Output sample rate is 24000 Hz (F5-TTS native). F5-TTS requires both the reference audio file and its transcript — unlike XTTS v2 which only needed the audio.

**`split_sentences()`** — splits on `[.!?]` followed by whitespace and an uppercase Finnish letter (`[A-ZÄÖÅ]`). This correctly preserves Finnish abbreviations (`mm.`, `em.`, `ao.` etc.) because they are always followed by lowercase letters.

## Key Constants

| Constant | File | Value | Note |
|---|---|---|---|
| `SAMPLE_RATE` | `1_prepare_voices.py` | 24000 | Reference voice WAV sample rate (F5-TTS expects 24 kHz) |
| `F5_SAMPLE_RATE` | `2_generate_audio.py` | 24000 | F5-TTS output sample rate |
| `SENTENCE_PAUSE_SEC` | `2_generate_audio.py` | 0.45 | Silence between synthesized sentences |

## Tech Stack

- **TTS model**: F5-TTS (flow matching, zero-shot voice cloning, multilingual)
- **Voice source**: Mozilla Common Voice 17.0 Finnish validated split
- **Audio I/O**: `soundfile` for WAV read/write, `librosa` for resampling
- **Language**: Python 3.11+, type hints throughout, `pathlib.Path` for all paths
- **Style**: snake_case, no external config files — constants live at module top level
