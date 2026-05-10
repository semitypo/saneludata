"""
Vaihe 1: Lataa Mozilla Common Voice Finnish ja rakenna referenssiäänitiedostot.

Common Voice on lokakuusta 2025 alkaen saatavilla vain suoraan Mozillalta:
  https://commonvoice.mozilla.org/fi/datasets

Lataa Finnish-datasetti (tar.gz), pura se ja anna polku --cv-path -argumentilla.

Käyttö:
  python 1_prepare_voices.py --cv-path /workspace/cv-corpus-21.0-2025-03-14/fi
  python 1_prepare_voices.py --cv-path /workspace/cv-corpus-21.0-2025-03-14/fi --speakers 30
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 24000
OUTPUT_DIR = Path("data/reference_voices")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-path", required=True,
                        help="Polku purettuun Common Voice fi-hakemistoon (sisältää validated.tsv ja clips/)")
    parser.add_argument("--speakers", type=int, default=20,
                        help="Valittavien puhujien määrä (oletus: 20)")
    parser.add_argument("--min-clips", type=int, default=8,
                        help="Vähimmäisklippimäärä per puhuja (oletus: 8)")
    parser.add_argument("--ref-duration", type=float, default=15.0,
                        help="Referenssiäänen tavoitekesto sekunteina (oletus: 15)")
    args = parser.parse_args()

    cv_path = Path(args.cv_path)
    tsv_path = cv_path / "validated.tsv"
    clips_dir = cv_path / "clips"

    if not tsv_path.exists():
        raise FileNotFoundError(f"validated.tsv ei löydy: {tsv_path}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"clips/-hakemisto ei löydy: {clips_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Ladataan metadata: {tsv_path}")
    clips = load_tsv(tsv_path, clips_dir)
    print(f"Löydettiin {len(clips)} validoitua nauhoitusta.")

    speakers: dict[str, list] = defaultdict(list)
    for clip in clips:
        speakers[clip["client_id"]].append(clip)

    print(f"Uniikkeja puhujia: {len(speakers)}")

    valid = {k: v for k, v in speakers.items() if len(v) >= args.min_clips}
    print(f"Puhujia joilla {args.min_clips}+ klippiä: {len(valid)}")

    if len(valid) == 0:
        raise RuntimeError("Ei löytynyt tarpeeksi puhujia. Laske --min-clips arvoa.")

    selected = select_diverse_speakers(valid, args.speakers)
    print(f"Valittiin {len(selected)} puhujaa.\n")

    manifest = {}

    for idx, (_, clips_list) in enumerate(
        tqdm(selected.items(), desc="Rakennetaan referenssiäänet")
    ):
        out_path = OUTPUT_DIR / f"speaker_{idx:03d}.wav"
        gender = clips_list[0].get("gender") or "unknown"
        age = clips_list[0].get("age") or "unknown"

        ref_text = create_reference_audio(clips_list, out_path, args.ref_duration)

        manifest[f"speaker_{idx:03d}"] = {
            "file": str(out_path),
            "ref_text": ref_text,
            "gender": gender,
            "age": age,
            "clip_count": len(clips_list),
        }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nValmis!")
    print(f"  Referenssiäänet: {OUTPUT_DIR}/speaker_000.wav ... speaker_{len(selected)-1:03d}.wav")
    print(f"  Manifesti:       {manifest_path}")
    print(f"\nSeuraava vaihe: python 2_generate_audio.py data/transcriptions/sanelut.csv")


def load_tsv(tsv_path: Path, clips_dir: Path) -> list[dict]:
    clips = []
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = clips_dir / row["path"]
            if audio_path.exists():
                row["audio_path"] = str(audio_path)
                clips.append(row)
    return clips


def select_diverse_speakers(speakers: dict, n: int) -> dict:
    sorted_all = sorted(speakers.items(), key=lambda x: len(x[1]), reverse=True)

    male = [(k, v) for k, v in sorted_all if v[0].get("gender") == "male"]
    female = [(k, v) for k, v in sorted_all if v[0].get("gender") == "female"]
    other = [(k, v) for k, v in sorted_all if v[0].get("gender") not in ("male", "female")]

    selected: dict = {}
    half = n // 2

    for k, v in male[:half]:
        selected[k] = v
    for k, v in female[:n - half]:
        selected[k] = v

    for k, v in other + sorted_all:
        if len(selected) >= n:
            break
        if k not in selected:
            selected[k] = v

    return selected


def create_reference_audio(clips: list, output_path: Path, target_duration: float) -> str:
    random.shuffle(clips)

    segments = []
    sentences = []
    total_sec = 0.0
    silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)

    for clip in clips:
        if total_sec >= target_duration:
            break

        try:
            arr, sr = librosa.load(clip["audio_path"], sr=None, mono=True)
        except Exception:
            continue

        if sr != SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)

        arr = arr.astype(np.float32)
        peak = np.max(np.abs(arr))
        if peak > 0:
            arr = arr / peak * 0.88

        sentence = (clip.get("sentence") or "").strip()
        if sentence:
            sentences.append(sentence)

        segments.append(arr)
        segments.append(silence)
        total_sec += len(arr) / SAMPLE_RATE

    if segments:
        combined = np.concatenate(segments)
        sf.write(output_path, combined, SAMPLE_RATE)

    return " ".join(sentences)


if __name__ == "__main__":
    main()
