"""
Step 0: Create Finnish Bark speaker presets.

Generates 10 distinct Finnish voice presets using Bark's own models
with different random seeds. Each preset produces a consistently
different-sounding speaker when used with generate_audio().

Usage:
  python 0_create_bark_presets.py
  python 0_create_bark_presets.py --speakers 10
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

FI_SEED_TEXTS = [
    "Tässä on radiologinen lausunto.",
    "Keuhkot ovat normaalit.",
    "Maksa on säännöllinen.",
    "Munuaiset kuvautuvat symmetrisinä.",
    "Ei merkkejä etäpesäkkeistä.",
    "Imusolmukkeet ovat normaalikokoiset.",
    "Virtsarakko on sileäseinäinen.",
    "Luusto on normaali.",
    "Verisuonisto on avoin.",
    "Tutkimus on normaali.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", type=int, default=10,
                        help="Number of presets to create (default: 10)")
    args = parser.parse_args()

    import bark
    bark_dir = Path(bark.__file__).parent
    prompts_dir = bark_dir / "assets" / "prompts" / "v2"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Creating {args.speakers} Finnish Bark speaker presets...")

    from bark.generation import generate_text_semantic, generate_coarse, generate_fine

    for idx in tqdm(range(args.speakers), desc="Creating presets"):
        preset_path = prompts_dir / f"fi_speaker_{idx}.npz"
        if preset_path.exists():
            tqdm.write(f"  Skipping (already exists): fi_speaker_{idx}.npz")
            continue

        seed = idx * 1000 + 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_text = FI_SEED_TEXTS[idx % len(FI_SEED_TEXTS)]

        try:
            semantic_tokens = generate_text_semantic(
                seed_text,
                temp=0.7,
                min_eos_p=0.05,
            )

            coarse_tokens = generate_coarse(
                semantic_tokens,
                temp=0.7,
            )

            fine_tokens = generate_fine(
                coarse_tokens,
                temp=0.5,
            )

            np.savez(
                str(preset_path),
                semantic_prompt=np.array(semantic_tokens, dtype=np.int32),
                coarse_prompt=np.array(coarse_tokens, dtype=np.int64),
                fine_prompt=np.array(fine_tokens, dtype=np.int64),
            )

            tqdm.write(f"  Saved: fi_speaker_{idx}.npz (seed={seed})")

        except Exception as e:
            tqdm.write(f"  ERROR (speaker {idx}): {e}")

    created = list(prompts_dir.glob("fi_speaker_*.npz"))
    print(f"\nDone! {len(created)} Finnish speaker presets saved.")
    print(f"  Directory: {prompts_dir}")
    print("\nNext step:")
    print("  python 2_generate_audio.py data/transcriptions/sanelut.csv")


if __name__ == "__main__":
    main()
