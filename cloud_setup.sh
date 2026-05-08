#!/bin/bash
# RunPod-ympäristön pikaasennus.
# PyTorch + CUDA ovat jo valmiina RunPodin PyTorch-templatessa.
# Aja: bash cloud_setup.sh

set -e

echo "=== Asennetaan riippuvuudet ==="
pip install f5-tts datasets soundfile librosa tqdm huggingface_hub -q

echo ""
echo "=== Tarkistetaan GPU ==="
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  VAROITUS: GPU ei näy!')
"

echo ""
echo "=== Luodaan hakemistorakenne ==="
mkdir -p data/transcriptions data/reference_voices data/output

echo ""
echo "=== Valmis! Seuraavat vaiheet: ==="
echo "  1. huggingface-cli login"
echo "  2. python 1_prepare_voices.py"
echo "  3. python 2_generate_audio.py data/transcriptions/sanelut.csv"
echo "  4. python 3_push_to_hub.py --repo omakäyttäjänimi/radiologia-audio"
