# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Finnish radiological TTS training data pipeline. Takes transcribed radiological dictations (CSV) and produces WAV audio files using Coqui XTTS v2 voice cloning from Mozilla Common Voice Finnish speakers. Output is used to train Finnish speech/language models on radiological vocabulary.

## Environment

- **OS**: Windows 11, shell: bash (WSL or Git Bash)
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) — always target CUDA, never CPU-only
- **Python**: 3.11+, scripts run from repo root

## API Keys (.env)

API-avaimet tallennetaan `.env`-tiedostoon projektin juuressa. Tiedosto on gitignoressa — ei koskaan commitoida.

```bash
cp .env.example .env
# Täytä avaimet:
#   MOZILLA_DC_API_KEY  — Mozilla Data Collective (Common Voice lataus)
#   HF_TOKEN            — HuggingFace (datasettien pushaaminen)
```

## Setup (paikallinen, kotikone)

Coqui TTS vaatii Python <3.12, joten käytetään conda-ympäristöä:

```bash
conda create -n tts-radio python=3.11 -y
conda activate tts-radio

# PyTorch must be installed with CUDA before other packages
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Cloud GPU (RunPod)

Kun ajetaan RunPodissa (PyTorch-template, RTX A5000 tai vastaava, ≥60 GB volume):

```bash
# 1. Kloonaa repo JupyterLabin terminaalissa
git clone https://github.com/semitypo/saneludata.git /workspace/saneludata
cd /workspace/saneludata

# 2. Asenna riippuvuudet volumelle (kerran — säilyy podin uudelleenkäynnistysten välillä)
bash cloud_setup.sh
export PYTHONPATH=/workspace/packages:$PYTHONPATH

# 3. Aseta API-avaimet
cp .env.example .env
nano .env   # täytä MOZILLA_DC_API_KEY ja HF_TOKEN

# 4. Lataa Common Voice Finnish
bash download_cv.sh

# 5. Rakenna referenssiäänet
python 1_prepare_voices.py --cv-path /workspace/cv-corpus/fi

# 6. Luo CSV ja generoi audio
python 2_generate_audio.py data/transcriptions/sanelut.csv

# 7. Siirrä data HuggingFace Hubiin
python 3_push_to_hub.py --repo käyttäjänimi/radiologia-audio
```

**Tärkeää:** `export PYTHONPATH` pitää asettaa joka uudessa terminaali-sessiossa (tai lähde `source ~/.bashrc`). `cloud_setup.sh` lisää sen `.bashrc`:hen automaattisesti.

CSV-tiedosto: luo suoraan terminaalissa tai lataa JupyterLabin Upload-napilla. `data/`-hakemisto on gitignoressa.

## Pipeline Commands

```bash
# Step 1 — run once: build reference voice files from local Common Voice download
python 1_prepare_voices.py --cv-path /workspace/cv-corpus/fi
python 1_prepare_voices.py --cv-path /workspace/cv-corpus/fi --speakers 30 --min-clips 5

# Step 2 — generate audio from transcriptions CSV
python 2_generate_audio.py data/transcriptions/sanelut.csv
python 2_generate_audio.py data/transcriptions/sanelut.csv --speakers 10 --seed 123
```

Step 2 is resumable: already-generated WAV files are skipped automatically.

## Data Layout

```
data/                     ← gitignoressa
  transcriptions/         ← input CSV files (id, text columns)
  reference_voices/       ← speaker_000.wav … speaker_N.wav + manifest.json
  output/                 ← generated WAVs named {id}_{speaker_id}.wav + metadata.csv

/workspace/               ← RunPod volume disk (pysyvä)
  packages/               ← pip-paketit (asennettu cloud_setup.sh:lla)
  cv-corpus/fi/           ← Common Voice Finnish (ladattu download_cv.sh:lla)
  cv-fi.tar.gz            ← Common Voice arkisto (latauksen jälkeen voi poistaa)
```

**Input CSV format** (`id`, `text` columns required, UTF-8):
```
id,text
lausunto_001,"Keuhkojen posteroanteriorinen röntgenkuva..."
```

**Output metadata.csv** columns: `id, audio_file, speaker_id, duration_sec, text`

## Architecture

Three scripts + one download helper:

**`download_cv.sh`**
Lukee `.env`-tiedostosta `MOZILLA_DC_API_KEY`:n, hakee presigned URL:n Mozilla Data Collectivesta ja lataa Common Voice Finnish tar.gz:n `/workspace/cv-fi.tar.gz`:hen. Purkaa arkiston `/workspace/cv-corpus/`:iin. Ohittaa latauksen jos tiedosto on jo olemassa.

**`1_prepare_voices.py`**
Reads Mozilla Common Voice Finnish from a locally downloaded tar.gz. Reads `validated.tsv` and `clips/*.mp3`, groups by `client_id`, filters by minimum clip count, selects N speakers with gender diversity, concatenates clips into per-speaker reference WAV files at 24000 Hz. Writes `data/reference_voices/manifest.json`.

**`2_generate_audio.py`**
Loads XTTS v2 via `TTS.api.TTS`. For each CSV row, picks a random speaker from the manifest, calls `split_sentences()` to break the text into sentence chunks, synthesizes each sentence separately with `tts.tts(text, speaker_wav, language="fi")`, then concatenates with 450 ms pauses. Output sample rate is 24000 Hz. Only the reference audio file is needed — no transcript required.

**`3_push_to_hub.py`**
Reads `metadata.csv`, loads WAVs, creates HuggingFace Dataset with Audio column, pushes to private HF Hub dataset.

**`split_sentences()`** — splits on `[.!?]` followed by whitespace and an uppercase Finnish letter (`[A-ZÄÖÅ]`). This correctly preserves Finnish abbreviations (`mm.`, `em.`, `ao.` etc.) because they are always followed by lowercase letters.

## Key Constants

| Constant | File | Value | Note |
|---|---|---|---|
| `SAMPLE_RATE` | `1_prepare_voices.py` | 24000 | Reference voice WAV sample rate |
| `XTTS_SAMPLE_RATE` | `2_generate_audio.py` | 24000 | XTTS v2 output sample rate |
| `SENTENCE_PAUSE_SEC` | `2_generate_audio.py` | 0.45 | Silence between synthesized sentences |

## Tech Stack

- **TTS model**: Coqui XTTS v2 (zero-shot voice cloning, explicit Finnish language support)
- **Voice source**: Mozilla Common Voice 25.0 Finnish validated split (Mozilla Data Collective)
- **Package installer**: `uv` (parallel downloads, 10-100x faster than pip)
- **Audio I/O**: `soundfile` for WAV read/write, `librosa` for resampling
- **Language**: Python 3.11+, type hints throughout, `pathlib.Path` for all paths
- **Style**: snake_case, no external config files — constants live at module top level
