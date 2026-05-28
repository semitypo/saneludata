#!/bin/bash
# RunPod-ympäristön pikaasennus.
# PyTorch + CUDA ovat jo valmiina RunPodin PyTorch-templatessa.
# Aja: bash cloud_setup.sh

set -e

PACKAGES_DIR=/workspace/packages

echo "=============================="
echo " GPU-tarkistus"
echo "=============================="
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU:  {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  VAROITUS: GPU ei näy!')
"

echo ""
echo "=============================="
echo " Asennetaan uv (nopea pip)"
echo "=============================="
pip install uv --quiet
echo "  uv OK"

echo ""
echo "=============================="
echo " Tarkistetaan onko TTS jo asennettu"
echo "=============================="
if PYTHONPATH=$PACKAGES_DIR python -c "import TTS" 2>/dev/null; then
    echo "  TTS on jo asennettu ($PACKAGES_DIR) — ohitetaan."
else
    echo "  Asennetaan TTS + riippuvuudet -> $PACKAGES_DIR"
    echo "  (rinnakkaislataus, näyttää nopeuden ja edistymisen)"
    echo ""
    mkdir -p $PACKAGES_DIR
    uv pip install TTS datasets soundfile librosa tqdm huggingface_hub \
        --target $PACKAGES_DIR \
        --system \
        --no-cache
fi

echo ""
echo "=============================="
echo " Asetetaan PYTHONPATH"
echo "=============================="
export PYTHONPATH=$PACKAGES_DIR:$PYTHONPATH
grep -qF "export PYTHONPATH=$PACKAGES_DIR" ~/.bashrc 2>/dev/null || \
    echo "export PYTHONPATH=$PACKAGES_DIR:\$PYTHONPATH" >> ~/.bashrc
echo "  PYTHONPATH=$PACKAGES_DIR"

echo ""
echo "=============================="
echo " Luodaan hakemistorakenne"
echo "=============================="
mkdir -p data/transcriptions data/reference_voices data/output
echo "  data/transcriptions/"
echo "  data/reference_voices/"
echo "  data/output/"

echo ""
echo "=============================="
echo " Valmis!"
echo "=============================="
echo "  Seuraavat vaiheet:"
echo "  1. hf auth login"
echo "  2. python 1_prepare_voices.py --cv-path /workspace/cv-corpus-XX.0/fi"
echo "  3. python 2_generate_audio.py data/transcriptions/sanelut.csv"
echo "  4. python 3_push_to_hub.py --repo käyttäjänimi/radiologia-audio"
