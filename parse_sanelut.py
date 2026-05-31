"""
Utility: parse vartalon TT sanelut.txt into data/transcriptions/sanelut.csv
"""

import csv
import re
import sys
from pathlib import Path

INPUT = Path("data/vartalon TT sanelut.txt")
OUTPUT = Path("data/transcriptions/sanelut.csv")


def clean_text(text: str) -> str:
    # Remove leading bullet dashes
    text = re.sub(r'^\s*[-–]\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{2,}', '\n', text)
    # Join lines into single string
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = ' '.join(lines)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()


def parse(src: Path) -> list[dict]:
    raw = src.read_text(encoding="utf-8")

    # Split on "Keissi N" headers
    parts = re.split(r'(?=Keissi\s+\d+)', raw, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    rows = []
    keissi_counts: dict[int, int] = {}

    for part in parts:
        header_match = re.match(r'Keissi\s+(\d+)', part, re.IGNORECASE)
        if not header_match:
            continue
        num = int(header_match.group(1))

        # Body text after the header line
        body = part[header_match.end():].strip()
        body = clean_text(body)

        if not body:
            continue

        # Handle duplicate keissi numbers
        keissi_counts[num] = keissi_counts.get(num, 0) + 1
        if keissi_counts[num] == 1:
            doc_id = f"vartalo_tt_{num:03d}"
        else:
            doc_id = f"vartalo_tt_{num:03d}{'bcde'[keissi_counts[num]-2]}"

        rows.append({"id": doc_id, "text": body})

    return rows


def main():
    if not INPUT.exists():
        print(f"ERROR: {INPUT} ei löydy")
        sys.exit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = parse(INPUT)

    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Kirjoitettu {len(rows)} sanelua → {OUTPUT}\n")
    for r in rows:
        preview = r['text'][:80].replace('\n', ' ')
        print(f"  {r['id']}: {preview}…")


if __name__ == "__main__":
    main()
