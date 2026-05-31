# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Finnish radiological TTS training data pipeline. Takes transcribed radiological dictations (CSV) and produces WAV audio files using Finnish-NLP/Chatterbox-Finnish voice cloning from Mozilla Common Voice Finnish speakers. Output is used to fine-tune Finnish ASR models on radiological vocabulary.

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

```bash
conda create -n tts-radio python=3.11 -y
conda activate tts-radio

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
git clone https://huggingface.co/Finnish-NLP/Chatterbox-Finnish /workspace/chatterbox-finnish
cd /workspace/chatterbox-finnish && python setup.py && cd -
pip install -r requirements.txt
```

## Cloud GPU (RunPod)

Kun ajetaan RunPodissa (PyTorch-template, RTX A5000 tai vastaava, ≥60 GB volume):

```bash
# 1. Kloonaa repo JupyterLabin terminaalissa
git clone https://github.com/semitypo/saneludata.git /workspace/saneludata
cd /workspace/saneludata

# 2. Asenna riippuvuudet — kloonaa Chatterbox, lataa base-mallit, asenna paketit
bash cloud_setup.sh

# 3. Aseta API-avaimet
cp .env.example .env
nano .env   # täytä MOZILLA_DC_API_KEY ja HF_TOKEN

# 4. Lataa Common Voice Finnish (upload JupyterLabilla tai .env:llä)
#    Tallenna arkisto: data/cv-fi.tar.gz
#    cloud_setup.sh purkaa sen automaattisesti

# 5. Rakenna referenssiäänet (ajetaan kerran)
python 1_prepare_voices.py --cv-path data/cv-corpus-25.0-2026-03-09/fi

# 6. Generoi audio (15-20 puhujaa)
python 2_generate_audio.py data/transcriptions/sanelut.csv --speakers 15

# 7. Siirrä data HuggingFace Hubiin
python 3_push_to_hub.py --repo käyttäjänimi/radiologia-audio
```

**Tärkeää:** `source ~/.bashrc` uudessa terminaali-sessiossa (cloud_setup.sh asettaa PYTHONPATH automaattisesti).

CSV-tiedosto: luo suoraan terminaalissa tai lataa JupyterLabin Upload-napilla. `data/`-hakemisto on gitignoressa.

## Pipeline Commands

```bash
# Step 1 — run once: build reference voice files from Common Voice download
python 1_prepare_voices.py --cv-path data/cv-corpus-25.0-2026-03-09/fi
python 1_prepare_voices.py --cv-path data/cv-corpus-25.0-2026-03-09/fi --speakers 20 --min-clips 5

# Step 2 — generate audio from transcriptions CSV (multi-speaker)
python 2_generate_audio.py data/transcriptions/sanelut.csv
python 2_generate_audio.py data/transcriptions/sanelut.csv --speakers 15 --seed 123
```

Step 2 is resumable: already-generated WAV files are skipped automatically.

## Data Layout

```
data/                     ← gitignoressa
  transcriptions/         ← input CSV files (id, text columns)
  cv-fi.tar.gz            ← Common Voice arkisto (puretaan cloud_setup.sh:lla)
  cv-corpus-25.0-.../fi/  ← purettu Common Voice Finnish
  reference_voices/       ← speaker_000.wav … speaker_N.wav + manifest.json
  output/                 ← generated WAVs named {id}_{speaker_id}.wav + metadata.csv

/workspace/               ← RunPod volume disk (pysyvä)
  packages/               ← pip-paketit (asennettu cloud_setup.sh:lla)
  chatterbox-finnish/     ← Finnish-NLP/Chatterbox-Finnish + base-mallit
```

**Input CSV format** (`id`, `text` columns required, UTF-8):
```
id,text
lausunto_001,"Keuhkojen posteroanteriorinen röntgenkuva..."
```

**Output metadata.csv** columns: `id, audio_file, speaker_id, duration_sec, text`

## Architecture

Three scripts:

**`1_prepare_voices.py`**
Reads Mozilla Common Voice Finnish (`validated.tsv` + `clips/*.mp3`), groups by `client_id`, filters by minimum clip count, selects N speakers with gender diversity, concatenates clips into per-speaker reference WAV files at 24000 Hz. Writes `data/reference_voices/manifest.json`.

**`2_generate_audio.py`**
Loads Finnish-NLP/Chatterbox-Finnish from `/workspace/chatterbox-finnish`. Injects Finnish fine-tuned weights (`best_finnish_multilingual_cp986.safetensors`). For each CSV row × each speaker in manifest, calls `split_sentences()` to chunk the text, synthesizes each sentence with `engine.generate(text, audio_prompt_path=speaker_wav, **FI_PARAMS)`, then concatenates with 450 ms pauses. Output sample rate from `engine.sr`. Output filename: `{id}_{speaker_id}.wav`.

**`3_push_to_hub.py`**
Reads `metadata.csv`, loads WAVs, creates HuggingFace Dataset with Audio column, pushes to private HF Hub dataset.

**`split_sentences()`** — splits on `[.!?]` followed by whitespace and an uppercase Finnish letter (`[A-ZÄÖÅ]`). Preserves Finnish abbreviations (`mm.`, `em.`, `ao.` etc.) which are always followed by lowercase letters.

## Dictation Style: Punctuation

Finnish radiologists dictate punctuation marks aloud. The `punctuation_to_spoken()` function converts all symbols to spoken words before TTS synthesis, matching real dictation style:

| Symbol | Spoken |
|--------|--------|
| `,` | pilkku |
| `.` | piste |
| `;` | puolipiste |
| `:` | kaksoispiste |
| `(` | sulku auki |
| `)` | sulku kiinni |
| `/` | kautta |
| `—` `–` | ajatusviiva |
| `1.3.2024` | ensimmäinen kolmatta kakstuhatta kaksikymmentäneljä |
| Roman numerals (`III` etc.) | roomalainen + spoken Finnish number (e.g. "roomalainen kolme") |

This means the ASR model learns to transcribe spoken punctuation back to symbols — the target output of the model includes punctuation in written form.

## Key Constants

| Constant | File | Value | Note |
|---|---|---|---|
| `SAMPLE_RATE` | `1_prepare_voices.py` | 24000 | Reference voice WAV sample rate |
| `SENTENCE_PAUSE_SEC` | `2_generate_audio.py` | 0.45 | Silence between synthesized sentences |
| `FI_PARAMS` | `2_generate_audio.py` | rep_penalty=1.5, temp=0.8, exag=0.5, cfg=0.3 | Optimized for Finnish phonology |

## Tech Stack

- **TTS model**: Finnish-NLP/Chatterbox-Finnish (zero-shot voice cloning, MOS 4.34, WER 2.76%)
- **Base model**: ResembleAI/chatterbox-turbo + Finnish fine-tune (`best_finnish_multilingual_cp986.safetensors`)
- **Voice source**: Mozilla Common Voice 25.0 Finnish validated split
- **Package installer**: `uv` (parallel downloads, 10-100x faster than pip)
- **Audio I/O**: `soundfile` for WAV read/write, `librosa` for resampling
- **Language**: Python 3.11+, type hints throughout, `pathlib.Path` for all paths
- **Style**: snake_case, no external config files — constants live at module top level
