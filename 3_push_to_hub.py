"""
Vaihe 3: Lataa generoitu äänidata HuggingFace Hubiin yksityisenä datasettinä.

Käyttö:
  python 3_push_to_hub.py --repo käyttäjänimi/radiologia-audio
  python 3_push_to_hub.py --repo käyttäjänimi/radiologia-audio --public
"""

import argparse
import csv
import sys
from pathlib import Path

OUTPUT_DIR = Path("data/output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo-nimi, esim. 'omanimi/radiologia-audio'")
    parser.add_argument("--public", action="store_true",
                        help="Julkinen datasetti (oletus: yksityinen)")
    args = parser.parse_args()

    metadata_path = OUTPUT_DIR / "metadata.csv"
    if not metadata_path.exists():
        print(f"VIRHE: {metadata_path} puuttuu. Aja ensin 2_generate_audio.py.")
        sys.exit(1)

    rows = load_metadata(metadata_path)
    print(f"Löydettiin {len(rows)} äänitiedostoa.")

    missing = [r for r in rows if not Path(r["audio_file"]).exists()]
    if missing:
        print(f"VAROITUS: {len(missing)} äänitiedostoa puuttuu — ohitetaan.")
        rows = [r for r in rows if Path(r["audio_file"]).exists()]

    print(f"Ladataan {len(rows)} tiedostoa HuggingFace Hubiin: {args.repo}")
    print("Tämä voi kestää pitkään riippuen aineiston koosta...\n")

    try:
        from datasets import Dataset, Audio
    except ImportError:
        print("VIRHE: pip install datasets")
        sys.exit(1)

    data = {
        "id": [r["id"] for r in rows],
        "audio": [r["audio_file"] for r in rows],
        "text": [r["text"] for r in rows],
        "speaker_id": [r["speaker_id"] for r in rows],
        "duration_sec": [float(r["duration_sec"]) for r in rows],
    }

    ds = Dataset.from_dict(data)
    ds = ds.cast_column("audio", Audio(sampling_rate=24000))

    privacy = "public" if args.public else "private"
    print(f"Ladataan ({privacy})...")

    ds.push_to_hub(
        args.repo,
        private=not args.public,
        commit_message=f"Add {len(rows)} Finnish radiological TTS samples",
    )

    total_min = sum(float(r["duration_sec"]) for r in rows) / 60
    print(f"\nValmis!")
    print(f"  Näytteitä:      {len(rows)}")
    print(f"  Kokonaiskesto:  {total_min:.1f} min ({total_min/60:.2f} h)")
    print(f"  HuggingFace:    https://huggingface.co/datasets/{args.repo}")


def load_metadata(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
