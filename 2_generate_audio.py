"""
Step 2: Generate audio from transcribed radiological dictations.

Uses Finnish-NLP/Chatterbox-Finnish (zero-shot voice cloning) with
speaker reference WAVs built by 1_prepare_voices.py.

Requires:
  git clone https://huggingface.co/Finnish-NLP/Chatterbox-Finnish /workspace/chatterbox-finnish
  cd /workspace/chatterbox-finnish && python setup.py
  (cloud_setup.sh handles all of this automatically)

Usage:
  python 2_generate_audio.py data/transcriptions/sanelut.csv
  python 2_generate_audio.py data/transcriptions/sanelut.csv --speakers 10 --seed 123
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

CHATTERBOX_DIR = Path("/workspace/chatterbox-finnish")
FINNISH_CHECKPOINT = CHATTERBOX_DIR / "models/best_finnish_multilingual_cp986.safetensors"
OUTPUT_DIR = Path("data/output")
MANIFEST_PATH = Path("data/reference_voices/manifest.json")
SENTENCE_PAUSE_SEC = 0.45

FI_PARAMS = dict(
    repetition_penalty=1.5,
    temperature=0.8,
    exaggeration=0.5,
    cfg_weight=0.3,
)

_DAY_ORDINALS = {
    1: 'ensimmäinen', 2: 'toinen', 3: 'kolmas', 4: 'neljäs',
    5: 'viides', 6: 'kuudes', 7: 'seitsemäs', 8: 'kahdeksas',
    9: 'yhdeksäs', 10: 'kymmenes', 11: 'yhdestoista',
    12: 'kahdestoista', 13: 'kolmastoista', 14: 'neljästoista',
    15: 'viidestoista', 16: 'kuudestoista', 17: 'seitsemästoista',
    18: 'kahdeksastoista', 19: 'yhdeksästoista', 20: 'kahdeskymmenes',
    21: 'kahdeskymmenesensimmäinen', 22: 'kahdeskymmenestoinen',
    23: 'kahdeskymmeneskolmas', 24: 'kahdeskymmenesneljäs',
    25: 'kahdeskymmenesviides', 26: 'kahdeskymmeneskuudes',
    27: 'kahdeskymmenesseitsemäs', 28: 'kahdeskymmeneskahdeksas',
    29: 'kahdeskymmenesyhdeksäs', 30: 'kolmaskymmenes',
    31: 'kolmaskymmenesensimmäinen',
}

_MONTH_PARTITIVES = {
    1: 'ensimmäistä', 2: 'toista', 3: 'kolmatta', 4: 'neljättä',
    5: 'viidettä', 6: 'kuudetta', 7: 'seitsemättä', 8: 'kahdeksatta',
    9: 'yhdeksättä', 10: 'kymmenettä', 11: 'yhdettätoista',
    12: 'kahdettatoista',
}

_MONTH_NAMES = {
    1: 'tammikuu', 2: 'helmikuu', 3: 'maaliskuu', 4: 'huhtikuu',
    5: 'toukokuu', 6: 'kesäkuu', 7: 'heinäkuu', 8: 'elokuu',
    9: 'syyskuu', 10: 'lokakuu', 11: 'marraskuu', 12: 'joulukuu',
}

ROMAN_NUMERALS = {
    'XII': 'roomalainen kaksitoista', 'XI': 'roomalainen yksitoista', 'X': 'roomalainen kymmenen',
    'IX': 'roomalainen yhdeksän', 'VIII': 'roomalainen kahdeksan', 'VII': 'roomalainen seitsemän',
    'VI': 'roomalainen kuusi', 'V': 'roomalainen viisi', 'IV': 'roomalainen neljä',
    'III': 'roomalainen kolme', 'II': 'roomalainen kaksi', 'I': 'roomalainen yksi',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="CSV with 'id' and 'text' columns")
    parser.add_argument("--speakers", type=int, default=20,
                        help="Max speakers from manifest (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.input_csv)
    speakers = load_manifest(args.speakers)
    print(f"Dictations: {len(rows)}  |  Speakers: {len(speakers)}  |  Total: {len(rows) * len(speakers)}")

    engine = load_engine()

    metadata_rows = []
    failed = []

    total = len(rows) * len(speakers)
    with tqdm(total=total, desc="Generating audio") as pbar:
        for row in rows:
            doc_id = row["id"].strip()
            text = row["text"].strip()

            if not text:
                pbar.update(len(speakers))
                continue

            for speaker_id, speaker_info in speakers.items():
                out_path = OUTPUT_DIR / f"{doc_id}_{speaker_id}.wav"

                if out_path.exists():
                    pbar.write(f"  Skipping: {out_path.name}")
                    pbar.update(1)
                    continue

                try:
                    wav, sr = synthesize_long_text(engine, text, speaker_info["file"])
                    sf.write(out_path, wav, sr)
                    duration_sec = len(wav) / sr
                    metadata_rows.append({
                        "id": doc_id,
                        "audio_file": str(out_path),
                        "speaker_id": speaker_id,
                        "duration_sec": round(duration_sec, 2),
                        "text": text,
                    })
                except Exception as e:
                    import traceback
                    pbar.write(f"  ERROR ({doc_id}/{speaker_id}): {e}")
                    pbar.write(traceback.format_exc())
                    failed.append(f"{doc_id}/{speaker_id}")

                pbar.update(1)

    write_metadata(metadata_rows, OUTPUT_DIR / "metadata.csv")

    total_min = sum(r["duration_sec"] for r in metadata_rows) / 60
    print(f"\nDone!")
    print(f"  Audio files:  {len(metadata_rows)}")
    print(f"  Total length: {total_min:.1f} min ({total_min/60:.2f} h)")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print(f"  Metadata:     {OUTPUT_DIR / 'metadata.csv'}")

    if failed:
        print(f"\n  Failed ({len(failed)}): {', '.join(failed[:10])}")


def load_engine():
    import torch

    if not CHATTERBOX_DIR.exists():
        raise RuntimeError(
            f"Chatterbox-Finnish not found: {CHATTERBOX_DIR}\n"
            "Run: git clone https://huggingface.co/Finnish-NLP/Chatterbox-Finnish /workspace/chatterbox-finnish"
        )

    pretrained_dir = CHATTERBOX_DIR / "pretrained_models"
    if not pretrained_dir.exists():
        raise RuntimeError(
            f"Pretrained models not found: {pretrained_dir}\n"
            f"Run: cd {CHATTERBOX_DIR} && python setup.py"
        )

    sys.path.insert(0, str(CHATTERBOX_DIR))
    from src.chatterbox_.tts import ChatterboxTTS
    from safetensors.torch import load_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Chatterbox engine ({device})...")
    engine = ChatterboxTTS.from_local(str(pretrained_dir), device=device)

    if FINNISH_CHECKPOINT.exists():
        print(f"Injecting Finnish weights: {FINNISH_CHECKPOINT.name}")
        checkpoint = load_file(str(FINNISH_CHECKPOINT))
        t3_state = {k[3:] if k.startswith("t3.") else k: v for k, v in checkpoint.items()}
        engine.t3.load_state_dict(t3_state, strict=False)
    else:
        print(f"WARNING: Finnish checkpoint not found: {FINNISH_CHECKPOINT}")
        print("         Running with base Chatterbox weights (Finnish quality may be reduced).")

    print("Model loaded.")
    return engine


def load_manifest(max_speakers: int) -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Speaker manifest not found: {MANIFEST_PATH}\n"
            "Run: python 1_prepare_voices.py --cv-path <path-to-cv-fi>"
        )
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    keys = list(manifest.keys())[:max_speakers]
    return {k: manifest[k] for k in keys}


def synthesize_long_text(engine, text: str, speaker_wav: str) -> tuple[np.ndarray, int]:
    sentences = split_sentences(text)
    sr = engine.sr
    pause = np.zeros(int(SENTENCE_PAUSE_SEC * sr), dtype=np.float32)
    segments = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        spoken = punctuation_to_spoken(sentence)
        wav_tensor = engine.generate(
            text=spoken,
            audio_prompt_path=speaker_wav,
            **FI_PARAMS,
        )
        audio = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
        segments.append(audio)
        segments.append(pause)

    return (np.concatenate(segments) if segments else np.zeros(sr, dtype=np.float32)), sr


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÅ])', text)
    return [p.strip() for p in parts if p.strip()]


def _colloquial_number(n: int) -> str:
    ones = ['', 'yks', 'kaks', 'kolme', 'neljä', 'viis', 'kuus',
            'seittemän', 'kaheksan', 'yheksän']
    teens = ['kymmenen', 'ykstoista', 'kakstoist', 'kolmetoista',
             'neljätoista', 'viistoista', 'kuustoista', 'seittemäntoista',
             'kaheksantoista', 'yheksäntoista']
    tens_words = ['', '', 'kakskyt', 'kolkyt', 'neljäkyt', 'viiskyt',
                  'kuuskyt', 'seittemänkyt', 'kaheksankyt', 'yheksänkyt']
    if n < 10:
        return ones[n]
    elif n < 20:
        return teens[n - 10]
    else:
        t = tens_words[n // 10]
        o = ones[n % 10]
        return (t + ' ' + o).strip() if o else t


def _month_year_to_spoken(match) -> str:
    m, y = int(match.group(1)), int(match.group(2))
    if m < 1 or m > 12:
        return match.group(0)
    month = _MONTH_NAMES[m]
    if y < 100:
        year = _colloquial_number(y)
    elif 2000 <= y <= 2099:
        rest = y - 2000
        year = ('kakstuhatta ' + _colloquial_number(rest)).strip() if rest else 'kakstuhatta'
    else:
        year = str(y)
    return f'{month} {year}'


def _date_to_spoken(match) -> str:
    d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
    day = _DAY_ORDINALS.get(d, str(d))
    month = _MONTH_PARTITIVES.get(m, str(m))
    rest = y - 2000
    year = ('kakstuhatta ' + _colloquial_number(rest)).strip() if 2000 <= y <= 2099 else str(y)
    return f'{day} {month} {year}'


def punctuation_to_spoken(text: str) -> str:
    # Full dates: 1.3.2024
    text = re.sub(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b', _date_to_spoken, text)
    # Month/year: 10/25, 9/2025, 5 / 20
    text = re.sub(r'\b(\d{1,2})\s*/\s*(\d{2,4})\b', _month_year_to_spoken, text)
    # TNM staging: T3N0, T2N1M0
    text = re.sub(
        r'\bT(\d+)N(\d+)(?:M(\d+))?\b',
        lambda m: 'T ' + m.group(1) + ' N ' + m.group(2) + (' M ' + m.group(3) if m.group(3) else ''),
        text,
    )
    # klo → kello
    text = re.sub(r'\bklo\b', 'kello', text, flags=re.IGNORECASE)
    # nro → numero
    text = re.sub(r'\bnro\b', 'numero', text, flags=re.IGNORECASE)
    # Roman numerals
    for roman, spoken in ROMAN_NUMERALS.items():
        text = re.sub(rf'\b{roman}\b', spoken, text)
    text = text.replace('—', ' ajatusviiva')
    text = text.replace('–', ' ajatusviiva')
    text = text.replace('(', ' sulku auki ')
    text = text.replace(')', ' sulku kiinni ')
    text = text.replace(';', ' puolipiste')
    text = text.replace(':', ' kaksoispiste')
    text = text.replace('/', ' kautta ')
    text = text.replace(',', ' pilkku')
    text = text.replace('.', ' piste')
    return re.sub(r' +', ' ', text).strip()


def load_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: CSV file not found: {path}")
        sys.exit(1)
    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if "id" not in reader.fieldnames or "text" not in reader.fieldnames:
        print("ERROR: CSV must have 'id' and 'text' columns.")
        sys.exit(1)
    return rows


def write_metadata(rows: list[dict], path: Path):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
