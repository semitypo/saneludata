#!/bin/bash
# RunPod environment setup.
# PyTorch + CUDA are already available in the RunPod PyTorch template.
# Usage: bash cloud_setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGES_DIR=/workspace/packages
CHATTERBOX_DIR=/workspace/chatterbox-finnish
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
echo " Chatterbox-Finnish setup"
echo "=============================="
if [ -d "$CHATTERBOX_DIR/.git" ]; then
    echo "  Chatterbox-Finnish already cloned — checking for updates..."
    git -C "$CHATTERBOX_DIR" pull --ff-only 2>/dev/null || echo "  (could not pull, continuing with existing version)"
else
    echo "  Cloning Finnish-NLP/Chatterbox-Finnish..."
    apt-get install -y git-lfs > /dev/null 2>&1
    git lfs install
    git clone https://huggingface.co/Finnish-NLP/Chatterbox-Finnish "$CHATTERBOX_DIR"
    echo "  Cloned: $CHATTERBOX_DIR"
fi

if [ ! -f "$CHATTERBOX_DIR/pretrained_models/ve.safetensors" ]; then
    echo "  Downloading pretrained base models..."
    cd "$CHATTERBOX_DIR"
    python setup.py
    cd "$SCRIPT_DIR"
    echo "  Pretrained models ready."
else
    echo "  Pretrained models already present."
fi

if [ ! -f "$CHATTERBOX_DIR/models/best_finnish_multilingual_cp986.safetensors" ]; then
    echo "  WARNING: Finnish checkpoint not found in $CHATTERBOX_DIR/models/"
    echo "           Check that git-lfs downloaded the .safetensors file correctly:"
    echo "           cd $CHATTERBOX_DIR && git lfs pull"
else
    echo "  Finnish checkpoint OK."
fi

echo ""
echo "=============================="
echo " Installing uv (fast pip)"
echo "=============================="
pip install uv --quiet
echo "  uv OK"

echo ""
echo "=============================="
echo " Installing Python packages"
echo "=============================="
if python -c "import safetensors, transformers, silero_vad" 2>/dev/null; then
    echo "  Packages already installed ($PACKAGES_DIR) — skipping."
else
    echo "  Installing packages -> $PACKAGES_DIR"
    mkdir -p $PACKAGES_DIR
    # Install Chatterbox runtime deps (torch/xformers come from RunPod template)
    uv pip install \
        safetensors \
        "transformers==4.46.3" \
        "diffusers==0.29.0" \
        "peft==0.17.1" \
        "torchao==0.6.1" \
        "resemble-perth==1.0.1" \
        "conformer==0.3.2" \
        "s3tokenizer==0.3.0" \
        "silero-vad==6.2.0" \
        "librosa==0.11.0" \
        "soundfile==0.13.1" \
        pyloudnorm \
        num2words \
        ffmpeg-python \
        pandas \
        huggingface_hub \
        "datasets>=2.14.0" \
        tqdm \
        omegaconf \
        hf_transfer \
        gdown \
        requests \
        --target $PACKAGES_DIR \
        --system \
        --no-cache
    # Remove any torch that got pulled in — use RunPod's CUDA-built version
    rm -rf $PACKAGES_DIR/torch $PACKAGES_DIR/torchaudio $PACKAGES_DIR/torchvision \
           $PACKAGES_DIR/triton $PACKAGES_DIR/nvidia*
    echo "  Packages installed."
fi

echo ""
echo "=============================="
echo " Setting PYTHONPATH"
echo "=============================="
export PYTHONPATH=$CHATTERBOX_DIR:$PACKAGES_DIR:$PYTHONPATH
for line in \
    "export PYTHONPATH=$CHATTERBOX_DIR:\$PYTHONPATH" \
    "export PYTHONPATH=$PACKAGES_DIR:\$PYTHONPATH"; do
    grep -qF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
done
echo "  PYTHONPATH includes $CHATTERBOX_DIR and $PACKAGES_DIR"

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
echo " Reference voices"
echo "=============================="
if [ -f "$SCRIPT_DIR/data/reference_voices/manifest.json" ]; then
    COUNT=$(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/data/reference_voices/manifest.json')); print(len(d))")
    echo "  Reference voices OK ($COUNT speakers)"
elif [ -f "$CV_DIR/validated.tsv" ]; then
    echo "  Building reference voices..."
    cd "$SCRIPT_DIR"
    python 1_prepare_voices.py --cv-path "$CV_DIR"
else
    echo "  WARNING: Run 1_prepare_voices.py after Common Voice is ready."
fi

echo ""
echo "=============================="
echo " Ready!"
echo "=============================="
echo "  Run: python 2_generate_audio.py data/transcriptions/sanelut.csv"
echo "  Run: python 2_generate_audio.py data/transcriptions/sanelut.csv --speakers 15"
