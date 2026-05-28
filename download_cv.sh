#!/bin/bash
# Lataa Mozilla Common Voice Finnish -datasetti.
# Vaatii .env-tiedoston projektin juuressa.
# Käyttö: bash download_cv.sh

set -e

ENV_FILE="$(dirname "$0")/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "VIRHE: .env-tiedostoa ei löydy."
    echo "Kopioi .env.example -> .env ja täytä API-avaimesi."
    exit 1
fi

source "$ENV_FILE"

if [ -z "$MOZILLA_DC_API_KEY" ] || [ "$MOZILLA_DC_API_KEY" = "your_api_key_here" ]; then
    echo "VIRHE: Aseta MOZILLA_DC_API_KEY .env-tiedostoon."
    exit 1
fi

OUTPUT=/workspace/cv-fi.tar.gz
EXTRACT_DIR=/workspace/cv-corpus

if [ -f "$OUTPUT" ]; then
    echo "Tiedosto $OUTPUT on jo olemassa — ohitetaan lataus."
else
    echo "Haetaan latauslinkki Mozilla Data Collectivesta..."
    RESPONSE=$(curl -s -X POST \
        "https://mozilladatacollective.com/api/datasets/${MOZILLA_DC_DATASET_ID}/download" \
        -H "Authorization: Bearer ${MOZILLA_DC_API_KEY}" \
        -H "Content-Type: application/json")

    DOWNLOAD_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['downloadUrl'])")

    if [ -z "$DOWNLOAD_URL" ] || [ "$DOWNLOAD_URL" = "None" ]; then
        echo "VIRHE: Latauslinkin haku epäonnistui. Tarkista API-avain."
        echo "Vastaus: $RESPONSE"
        exit 1
    fi

    echo "Ladataan Common Voice Finnish -> $OUTPUT"
    wget -O "$OUTPUT" "$DOWNLOAD_URL" --progress=bar:force 2>&1
fi

echo ""
echo "Puretaan arkisto -> $EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
tar -xzf "$OUTPUT" -C "$EXTRACT_DIR" --checkpoint=1000 --checkpoint-action=echo="%T"

echo ""
echo "Valmis! Aja seuraavaksi:"
echo "  python 1_prepare_voices.py --cv-path $EXTRACT_DIR/fi"
