"""
Vaihe 2: Tuottaa äänitiedostoja litteroiduista radiologisista saneluista.

Käyttää XTTS v2 -mallia zero-shot ääniklonaukseen suomeksi.

Syöte (CSV):
  id,text
  lausunto_001,"Keuhkojen posteroanteriorinen röntgenkuva..."

Käyttö:
  python 2_generate_audio.py data/transcriptions/sanelut.csv
  python 2_generate_audio.py data/transcriptions/sanelut.csv --speakers 10 --seed 123
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

VOICES_DIR = Path("data/reference_voices")
OUTPUT_DIR = Path("data/output")

XTTS_SAMPLE_RATE = 24000
SENTENCE_PAUSE_SEC = 0.45

os.environ["COQUI_TOS_AGREED"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="CSV-tiedosto (sarakkeet: id, text)")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Käytettävien puhujien määrä (oletus: kaikki)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Satunnaisluvun siemen toistettavuuteen (oletus: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    speakers = load_speakers(args.speakers)
    rows = load_csv(args.input_csv)

    print(f"Puhujia:  {len(speakers)}")
    print(f"Saneluja: {len(rows)}")

    print("\nLadataan XTTS v2 -malli (ensimmäisellä kerralla ~2 GB lataus)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Laite: {device}")
    if device == "cpu":
        print("VAROITUS: GPU ei löydy. CPU-ajo on erittäin hidas suurille aineistoille.")

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    metadata_rows = []
    failed = []

    for row in tqdm(rows, desc="Tuotetaan ääntä"):
        doc_id = row["id"].strip()
        text = row["text"].strip()

        if not text:
            continue

        speaker = random.choice(speakers)
        speaker_id = Path(speaker["file"]).stem
        out_path = OUTPUT_DIR / f"{doc_id}_{speaker_id}.wav"

        if out_path.exists():
            tqdm.write(f"  Ohitetaan (jo olemassa): {out_path.name}")
            continue

        try:
            wav = synthesize_long_text(tts, text, speaker["file"])
            sf.write(out_path, wav, XTTS_SAMPLE_RATE)
            duration_sec = len(wav) / XTTS_SAMPLE_RATE

            metadata_rows.append({
                "id": doc_id,
                "audio_file": str(out_path),
                "speaker_id": speaker_id,
                "duration_sec": round(duration_sec, 2),
                "text": text,
            })

        except Exception as e:
            tqdm.write(f"  VIRHE ({doc_id}): {e}")
            failed.append(doc_id)

    write_metadata(metadata_rows, OUTPUT_DIR / "metadata.csv")

    total_min = sum(r["duration_sec"] for r in metadata_rows) / 60
    print(f"\nValmis!")
    print(f"  Äänitiedostoja: {len(metadata_rows)}")
    print(f"  Kokonaiskesto:  {total_min:.1f} min ({total_min/60:.2f} h)")
    print(f"  Hakemisto:      {OUTPUT_DIR}")
    print(f"  Metatiedot:     {OUTPUT_DIR / 'metadata.csv'}")

    if failed:
        print(f"\n  Epäonnistuneet ({len(failed)}): {', '.join(failed[:10])}")


def synthesize_long_text(tts, text: str, ref_file: str) -> np.ndarray:
    sentences = split_sentences(text)
    pause = np.zeros(int(SENTENCE_PAUSE_SEC * XTTS_SAMPLE_RATE), dtype=np.float32)
    segments = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        spoken = punctuation_to_spoken(sentence)
        wav = tts.tts(text=spoken, speaker_wav=ref_file, language="fi")
        segments.append(np.array(wav, dtype=np.float32))
        segments.append(pause)

    return np.concatenate(segments) if segments else np.zeros(XTTS_SAMPLE_RATE, dtype=np.float32)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÅ])', text)
    return [p.strip() for p in parts if p.strip()]


ROMAN_NUMERALS = {
    'XII': 'kaksitoista', 'XI': 'yksitoista', 'X': 'kymmenen',
    'IX': 'yhdeksän', 'VIII': 'kahdeksan', 'VII': 'seitsemän',
    'VI': 'kuusi', 'V': 'viisi', 'IV': 'neljä',
    'III': 'kolme', 'II': 'kaksi', 'I': 'yksi',
}

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

def _date_to_spoken(match) -> str:
    d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
    day = _DAY_ORDINALS.get(d, str(d))
    month = _MONTH_PARTITIVES.get(m, str(m))
    rest = y - 2000
    year = ('kakstuhatta ' + _colloquial_number(rest)).strip() if 2000 <= y <= 2099 else str(y)
    return f'{day} {month} {year}'

def punctuation_to_spoken(text: str) -> str:
    """Muuntaa päivämäärät, välimerkit ja roomalaiset numerot puhutuiksi sanoiksi."""
    text = re.sub(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b', _date_to_spoken, text)
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


def load_speakers(n: int | None) -> list[dict]:
    manifest_path = VOICES_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"VIRHE: Referenssiääniä ei löydy. Aja ensin: python 1_prepare_voices.py")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    speakers = list(manifest.values())
    if n:
        speakers = speakers[:n]

    missing = [s["file"] for s in speakers if not Path(s["file"]).exists()]
    if missing:
        print(f"VIRHE: {len(missing)} referenssiäänitiedostoa puuttuu.")
        sys.exit(1)

    return speakers


def load_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"VIRHE: CSV-tiedostoa ei löydy: {path}")
        sys.exit(1)

    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if "id" not in reader.fieldnames or "text" not in reader.fieldnames:
        print("VIRHE: CSV:ssä täytyy olla sarakkeet 'id' ja 'text'.")
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
