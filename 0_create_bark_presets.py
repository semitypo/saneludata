"""
Step 0: Create Bark speaker presets from Mozilla Common Voice Finnish audio.

Encodes reference voices through Bark's EnCodec model and saves them as
.npz files at bark/assets/prompts/v2/fi_speaker_X.npz

Run 1_prepare_voices.py first to build the reference voices.

Usage:
  python 0_create_bark_presets.py
  python 0_create_bark_presets.py --speakers 10
"""

import argparse
import json
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from tqdm import tqdm

VOICES_DIR = Path("data/reference_voices")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", type=int, default=10,
                        help="Luotavien presettien määrä (oletus: 10)")
    args = parser.parse_args()

    import bark
    bark_dir = Path(bark.__file__).parent
    prompts_dir = bark_dir / "assets" / "prompts" / "v2"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = VOICES_DIR / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: Run 1_prepare_voices.py first.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    speakers = list(manifest.items())[:args.speakers]
    print(f"Creating {len(speakers)} Finnish Bark speaker presets from reference audio...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from bark.generation import load_codec_model, generate_text_semantic
    CODEC_MODEL_SAMPLE_RATE = 24000
    codec_model = load_codec_model(use_gpu=(device == "cuda"))

    # Lyhyt suomenkielinen lause semanttisten tokenien generointiin
    fi_seed_text = "Tässä on radiologinen lausunto potilaasta."

    for idx, (speaker_id, speaker_info) in enumerate(
        tqdm(speakers, desc="Luodaan presettejä")
    ):
        preset_path = prompts_dir / f"fi_speaker_{idx}.npz"
        if preset_path.exists():
            tqdm.write(f"  Skipping (already exists): fi_speaker_{idx}.npz")
            continue

        audio_path = Path(speaker_info["file"])
        if not audio_path.exists():
            tqdm.write(f"  MISSING: {audio_path}")
            continue

        try:
            # Lataa referenssiääni
            audio, sr = sf.read(str(audio_path))
            audio = torch.tensor(audio, dtype=torch.float32)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio = audio.unsqueeze(0)

            # Resample jos tarvitaan
            if sr != CODEC_MODEL_SAMPLE_RATE:
                import torchaudio
                audio = torchaudio.functional.resample(
                    audio, sr, CODEC_MODEL_SAMPLE_RATE
                )

            # Varmista mono
            if audio.shape[1] > 1:
                audio = audio.mean(dim=1, keepdim=True)

            audio = audio.to(device)

            # Enkoodaa EnCodecilla akustisiksi tokeneiksi
            with torch.no_grad():
                encoded_frames = codec_model.encode(audio)

            codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
            codes = codes.squeeze(0)  # (n_codebooks, T)

            fine_prompt = codes.cpu().numpy()
            coarse_prompt = codes[:2].cpu().numpy()

            # Generoi semanttiset tokenit suomenkielisestä tekstistä
            torch.manual_seed(idx * 42)
            semantic_tokens = generate_text_semantic(
                fi_seed_text,
                temp=0.7,
                min_eos_p=0.05,
            )

            np.savez(
                str(preset_path),
                semantic_prompt=semantic_tokens,
                coarse_prompt=coarse_prompt,
                fine_prompt=fine_prompt,
            )

            tqdm.write(f"  Saved: fi_speaker_{idx}.npz "
                       f"({speaker_info.get('gender','?')}, {speaker_info.get('age','?')})")

        except Exception as e:
            tqdm.write(f"  ERROR ({speaker_id}): {e}")

    created = list(prompts_dir.glob("fi_speaker_*.npz"))
    print(f"\nDone! {len(created)} Finnish speaker presets saved.")
    print(f"  Directory: {prompts_dir}")
    print("\nNext step:")
    print("  python 2_generate_audio.py data/transcriptions/sanelut.csv")


if __name__ == "__main__":
    main()
