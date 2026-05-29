#!/bin/bash
# RunPod-ympäristön pikaasennus.
# PyTorch + CUDA ovat jo valmiina RunPodin PyTorch-templatessa.
# Aja: bash cloud_setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGES_DIR=/workspace/packages
CV_ARCHIVE=$SCRIPT_DIR/data/cv-fi.tar.gz
CV_DIR=$SCRIPT_DIR/data/cv-corpus-25.0-2026-03-09/fi

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
echo " Common Voice Finnish"
echo "=============================="
if [ -f "$CV_DIR/validated.tsv" ]; then
    echo "  Common Voice on jo purettu — ohitetaan."
elif [ -f "$CV_ARCHIVE" ]; then
    echo "  Puretaan $CV_ARCHIVE ..."
    tar -xzf "$CV_ARCHIVE" -C "$SCRIPT_DIR/data/" --no-same-owner
    echo "  Purettu: $CV_DIR"
elif [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
    if [ -n "$MOZILLA_DC_API_KEY" ] && [ "$MOZILLA_DC_API_KEY" != "your_api_key_here" ]; then
        echo "  Ladataan Mozilla Data Collectivesta..."
        RESPONSE=$(curl -s -X POST \
            "https://mozilladatacollective.com/api/datasets/${MOZILLA_DC_DATASET_ID}/download" \
            -H "Authorization: Bearer ${MOZILLA_DC_API_KEY}" \
            -H "Content-Type: application/json")
        DOWNLOAD_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['downloadUrl'])" 2>/dev/null || echo "")
        if [ -n "$DOWNLOAD_URL" ] && [ "$DOWNLOAD_URL" != "None" ]; then
            wget -O "$CV_ARCHIVE" "$DOWNLOAD_URL" --progress=bar:force 2>&1
            echo "  Puretaan..."
            tar -xzf "$CV_ARCHIVE" -C "$SCRIPT_DIR/data/" --no-same-owner
            echo "  Valmis: $CV_DIR"
        else
            echo "  VAROITUS: Lataus epäonnistui. Lataa manuaalisesti ja siirrä:"
            echo "    data/cv-fi.tar.gz"
        fi
    else
        echo "  VAROITUS: Lisää MOZILLA_DC_API_KEY tiedostoon .env"
    fi
else
    echo "  VAROITUS: Common Voice puuttuu. Uploadaa JupyterLabissa:"
    echo "    data/cv-fi.tar.gz  (tiedostonimi tärkeä)"
fi

echo ""
echo "=============================="
echo " Tarkistetaan referenssiäänet"
echo "=============================="
if [ -f "$SCRIPT_DIR/data/reference_voices/manifest.json" ]; then
    COUNT=$(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/data/reference_voices/manifest.json')); print(len(d))")
    echo "  Referenssiäänet OK ($COUNT puhujaa)"
elif [ -f "$CV_DIR/validated.tsv" ]; then
    echo "  Rakennetaan referenssiäänet..."
    cd "$SCRIPT_DIR"
    python 1_prepare_voices.py --cv-path "$CV_DIR"
else
    echo "  VAROITUS: Aja 1_prepare_voices.py kun Common Voice on ladattu."
fi

echo ""
echo "=============================="
echo " Valmis!"
echo "=============================="
echo "  Aja: python 2_generate_audio.py data/transcriptions/sanelut.csv"
