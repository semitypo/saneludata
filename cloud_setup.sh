#!/bin/bash
# RunPod environment setup.
# PyTorch + CUDA are already available in the RunPod PyTorch template.
# Usage: bash cloud_setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGES_DIR=/workspace/packages
CV_ARCHIVE=$SCRIPT_DIR/data/cv-fi.tar.gz
CV_DIR=$SCRIPT_DIR/data/cv-corpus-25.0-2026-03-09/fi

echo "=============================="
echo " GPU check"
echo "=============================="
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU:  {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU found!')
"

echo ""
echo "=============================="
echo " Installing uv (fast pip)"
echo "=============================="
pip install uv --quiet
echo "  uv OK"

echo ""
echo "=============================="
echo " Checking if packages are installed"
echo "=============================="
if PYTHONPATH=$PACKAGES_DIR python -c "import bark" 2>/dev/null; then
    echo "  Packages already installed ($PACKAGES_DIR) — skipping."
else
    echo "  Installing packages -> $PACKAGES_DIR"
    echo "  (parallel downloads with speed and progress display)"
    echo ""
    mkdir -p $PACKAGES_DIR
    uv pip install piper-tts datasets soundfile librosa tqdm huggingface_hub \
        --target $PACKAGES_DIR \
        --system \
        --no-cache
    # Remove torch from workspace packages — use system CUDA-compatible version
    rm -rf $PACKAGES_DIR/torch $PACKAGES_DIR/torchaudio $PACKAGES_DIR/triton $PACKAGES_DIR/nvidia*
fi

echo ""
echo "=============================="
echo " Setting PYTHONPATH"
echo "=============================="
export PYTHONPATH=$PACKAGES_DIR:$PYTHONPATH
grep -qF "export PYTHONPATH=$PACKAGES_DIR" ~/.bashrc 2>/dev/null || \
    echo "export PYTHONPATH=$PACKAGES_DIR:\$PYTHONPATH" >> ~/.bashrc
echo "  PYTHONPATH=$PACKAGES_DIR"

echo ""
echo "=============================="
echo " Creating directory structure"
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
    echo "  Common Voice already extracted — skipping."
elif [ -f "$CV_ARCHIVE" ]; then
    echo "  Extracting $CV_ARCHIVE ..."
    tar -xzf "$CV_ARCHIVE" -C "$SCRIPT_DIR/data/" --no-same-owner
    echo "  Extracted: $CV_DIR"
elif [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
    if [ -n "$MOZILLA_DC_API_KEY" ] && [ "$MOZILLA_DC_API_KEY" != "your_api_key_here" ]; then
        echo "  Downloading from Mozilla Data Collective..."
        RESPONSE=$(curl -s -X POST \
            "https://mozilladatacollective.com/api/datasets/${MOZILLA_DC_DATASET_ID}/download" \
            -H "Authorization: Bearer ${MOZILLA_DC_API_KEY}" \
            -H "Content-Type: application/json")
        DOWNLOAD_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['downloadUrl'])" 2>/dev/null || echo "")
        if [ -n "$DOWNLOAD_URL" ] && [ "$DOWNLOAD_URL" != "None" ]; then
            wget -O "$CV_ARCHIVE" "$DOWNLOAD_URL" --progress=bar:force 2>&1
            echo "  Extracting..."
            tar -xzf "$CV_ARCHIVE" -C "$SCRIPT_DIR/data/" --no-same-owner
            echo "  Done: $CV_DIR"
        else
            echo "  WARNING: Download failed. Upload manually as: data/cv-fi.tar.gz"
        fi
    else
        echo "  WARNING: Set MOZILLA_DC_API_KEY in .env file"
    fi
else
    echo "  WARNING: Common Voice not found. Upload via JupyterLab as: data/cv-fi.tar.gz"
fi

echo ""
echo "=============================="
echo " Checking reference voices"
echo "=============================="
if [ -f "$SCRIPT_DIR/data/reference_voices/manifest.json" ]; then
    COUNT=$(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/data/reference_voices/manifest.json')); print(len(d))")
    echo "  Reference voices OK ($COUNT speakers)"
elif [ -f "$CV_DIR/validated.tsv" ]; then
    echo "  Building reference voices..."
    cd "$SCRIPT_DIR"
    python 1_prepare_voices.py --cv-path "$CV_DIR"
    echo "  Building Bark speaker presets..."
    python 0_create_bark_presets.py
else
    echo "  WARNING: Run 1_prepare_voices.py and 0_create_bark_presets.py after Common Voice is ready."
fi

echo ""
echo "=============================="
echo " Ready!"
echo "=============================="
echo "  Run: python 2_generate_audio.py data/transcriptions/sanelut.csv"
