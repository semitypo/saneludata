"""
Vaihe 1: Lataa Mozilla Common Voice Finnish ja rakenna referenssiäänitiedostot.

F5-TTS vaatii referenssiäänelle sekä audiotiedoston että litteraatin.
Molemmat tallennetaan manifest.json-tiedostoon.

Vaatimukset:
  - HuggingFace-tili osoitteessa huggingface.co
  - Hyväksy Common Voice -lisenssi dataset-sivulla:
    https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
  - Kirjaudu sisään: huggingface-cli login
    tai aseta ympäristömuuttuja: set HF_TOKEN=hf_xxxxx

Käyttö:
  python 1_prepare_voices.py
  python 1_prepare_voices.py --speakers 30 --min-clips 5
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

SAMPLE_RATE = 24000  # F5-TTS odottaa 24 kHz referenssiääntä
OUTPUT_DIR = Path("data/reference_voices")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", type=int, default=20,
                        help="Valittavien puhujien määrä (oletus: 20)")
    parser.add_argument("--min-clips", type=int, default=8,
                        help="Vähimmäisklippimäärä per puhuja (oletus: 8)")
    parser.add_argument("--ref-duration", type=float, default=15.0,
                        help="Referenssiäänen tavoitekesto sekunteina (oletus: 15)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Ladataan Mozilla Common Voice Finnish (validated)...")
    print("HUOM: Ensimmäisellä kerralla lataus voi kestää useita minuutteja (~2 GB).\n")

    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "fi",
        split="validated",
        trust_remote_code=True,
    )

    print(f"Löydettiin {len(ds)} validoitua nauhoitusta.")

    speakers: dict[str, list] = defaultdict(list)
    for item in tqdm(ds, desc="Ryhmitellään puhujittain"):
        speakers[item["client_id"]].append(item)

    print(f"Uniikkeja puhujia: {len(speakers)}")

    valid = {k: v for k, v in speakers.items() if len(v) >= args.min_clips}
    print(f"Puhujia joilla {args.min_clips}+ klippiä: {len(valid)}")

    if len(valid) == 0:
        raise RuntimeError("Ei löytynyt tarpeeksi puhujia. Laske --min-clips arvoa.")

    selected = select_diverse_speakers(valid, args.speakers)
    print(f"Valittiin {len(selected)} puhujaa (sukupuolidiversiteetti huomioitu).\n")

    manifest = {}

    for idx, (_, clips) in enumerate(
        tqdm(selected.items(), desc="Rakennetaan referenssiäänet")
    ):
        out_path = OUTPUT_DIR / f"speaker_{idx:03d}.wav"
        gender = clips[0].get("gender") or "unknown"
        age = clips[0].get("age") or "unknown"

        ref_text = create_reference_audio(clips, out_path, args.ref_duration)

        manifest[f"speaker_{idx:03d}"] = {
            "file": str(out_path),
            "ref_text": ref_text,
            "gender": gender,
            "age": age,
            "clip_count": len(clips),
        }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nValmis!")
    print(f"  Referenssiäänet: {OUTPUT_DIR}/speaker_000.wav ... speaker_{len(selected)-1:03d}.wav")
    print(f"  Manifesti:       {manifest_path}")
    print(f"\nSeuraava vaihe: python 2_generate_audio.py data/transcriptions/sanelut.csv")


def select_diverse_speakers(speakers: dict, n: int) -> dict:
    sorted_all = sorted(speakers.items(), key=lambda x: len(x[1]), reverse=True)

    male = [(k, v) for k, v in sorted_all if v[0].get("gender") == "male"]
    female = [(k, v) for k, v in sorted_all if v[0].get("gender") == "female"]
    other = [(k, v) for k, v in sorted_all if v[0].get("gender") not in ("male", "female")]

    selected: dict = {}
    half = n // 2

    for k, v in male[:half]:
        selected[k] = v
    for k, v in female[: n - half]:
        selected[k] = v

    for k, v in other + sorted_all:
        if len(selected) >= n:
            break
        if k not in selected:
            selected[k] = v

    return selected


def create_reference_audio(clips: list, output_path: Path, target_duration: float) -> str:
    """
    Yhdistää klippejä referenssiääneksi ja palauttaa käytettyjen klippien litteraatin.
    F5-TTS tarvitsee litteraatin ääniä vastaavan tekstin.
    """
    random.shuffle(clips)

    segments = []
    sentences = []
    total_sec = 0.0
    silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)

    for clip in clips:
        if total_sec >= target_duration:
            break

        arr = np.array(clip["audio"]["array"], dtype=np.float32)
        sr = clip["audio"]["sampling_rate"]

        if sr != SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)

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
