"""
Step 1: Build reference voice files from Mozilla Common Voice Finnish.

Download the Finnish dataset (tar.gz) from:
  https://commonvoice.mozilla.org/fi/datasets

Extract it and pass the path with --cv-path.

Usage:
  python 1_prepare_voices.py --cv-path /workspace/cv-corpus-25.0/fi
  python 1_prepare_voices.py --cv-path /workspace/cv-corpus-25.0/fi --speakers 30
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
                        help="Path to extracted Common Voice fi directory (contains validated.tsv and clips/)")
    parser.add_argument("--speakers", type=int, default=20,
                        help="Number of speakers to select (default: 20)")
    parser.add_argument("--min-clips", type=int, default=8,
                        help="Minimum clips per speaker (default: 8)")
    parser.add_argument("--ref-duration", type=float, default=15.0,
                        help="Target reference audio duration in seconds (default: 15)")
    args = parser.parse_args()

    cv_path = Path(args.cv_path)
    tsv_path = cv_path / "validated.tsv"
    clips_dir = cv_path / "clips"

    if not tsv_path.exists():
        raise FileNotFoundError(f"validated.tsv not found: {tsv_path}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"clips/ directory not found: {clips_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata: {tsv_path}")
    clips = load_tsv(tsv_path, clips_dir)
    print(f"Found {len(clips)} validated recordings.")

    speakers: dict[str, list] = defaultdict(list)
    for clip in clips:
        speakers[clip["client_id"]].append(clip)

    print(f"Unique speakers: {len(speakers)}")

    valid = {k: v for k, v in speakers.items() if len(v) >= args.min_clips}
    print(f"Speakers with {args.min_clips}+ clips: {len(valid)}")

    if len(valid) == 0:
        raise RuntimeError("No speakers found. Try lowering --min-clips.")

    selected = select_diverse_speakers(valid, args.speakers)
    print(f"Selected {len(selected)} speakers.\n")

    manifest = {}

    for idx, (_, clips_list) in enumerate(
        tqdm(selected.items(), desc="Building reference voices")
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

    print(f"\nDone!")
    print(f"  Reference voices: {OUTPUT_DIR}/speaker_000.wav ... speaker_{len(selected)-1:03d}.wav")
    print(f"  Manifest:         {manifest_path}")
    print(f"\nNext step: python 0_create_bark_presets.py")


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
