#!/usr/bin/env python3
"""
EasyOCR - Extract text from images (JPG/PNG) and PDFs
Usage: python easyocr_ocr.py <file_path> [--lang ar fr en] [--threshold 0.4] [--raw]
Install: pip install easyocr pdf2image pillow
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

try:
    import easyocr
except ImportError:
    os.system(f"{sys.executable} -m pip install easyocr")
    import easyocr

try:
    from PIL import Image
except ImportError:
    os.system(f"{sys.executable} -m pip install pillow")
    from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    os.system(f"{sys.executable} -m pip install pdf2image")
    from pdf2image import convert_from_path

DEFAULT_THRESHOLD = 0.4


# ---------------- OCR ---------------- #
def run_ocr_on_image(image_path: str, langs: list) -> list:
    """Run EasyOCR on an image file. Returns list of {word, confidence}."""
    reader = easyocr.Reader(langs, gpu=False)
    results = reader.readtext(image_path, detail=1, paragraph=False)

    words = []
    for (bbox, text, confidence) in results:
        text = text.strip()
        if not text:
            continue
        words.append({"word": text, "confidence": round(confidence, 4)})

    return words


def run_ocr(file_path: str, langs: list) -> list:
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    all_words = []

    if path.suffix.lower() == ".pdf":
        print("📄 Converting PDF pages to images (dpi=200)...")
        images = convert_from_path(str(path), dpi=200)
        for i, img in enumerate(images):
            print(f"  🔍 OCR on page {i+1}/{len(images)}...", end=" ", flush=True)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp.name, "JPEG")
                page_words = run_ocr_on_image(tmp.name, langs)
                os.unlink(tmp.name)
            print(f"{len(page_words)} words")
            all_words.extend(page_words)
    else:
        print(f"🔍 Running EasyOCR on {path.name} (langs={'+'.join(langs)})...")
        print("   (First run downloads models ~500MB — cached after)\n")
        all_words = run_ocr_on_image(str(path), langs)

    return all_words


# ---------------- CLEANING ---------------- #
def clean_ocr(ocr_json: list, threshold: float):
    clean_words, raw_words = [], []
    hallucinated = 0
    total = 0

    for w in ocr_json:
        total += 1
        raw_words.append(f"{w['word']}({w['confidence']:.2f})")
        if w["confidence"] < threshold:
            clean_words.append("[UNK]")
            hallucinated += 1
        else:
            clean_words.append(w["word"])

    return " ".join(clean_words), " ".join(raw_words), hallucinated, total


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EasyOCR text extractor — Arabic, French, English")
    parser.add_argument("input", help="Image or PDF file")
    parser.add_argument("--lang", "-l", nargs="+", default=["ar", "fr", "en"],
                        help="Languages to detect (default: ar fr en). "
                             "Examples: --lang ar   --lang fr en   --lang ar fr en")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Confidence threshold 0.0-1.0 (default: {DEFAULT_THRESHOLD}). "
                              "Use 0.0 to see ALL detected text.")
    parser.add_argument("--raw", action="store_true",
                        help="Also print raw words with confidence scores")
    parser.add_argument("--output", "-o", help="Save clean text to this file")
    args = parser.parse_args()

    ocr_json = run_ocr(args.input, args.lang)

    if not ocr_json:
        print("\n⚠️  No text detected at all.")
        print("Tips:")
        print("  • Try lowering threshold  : --threshold 0.0")
        print("  • Make sure image is clear and well-lit")
        print("  • Try specific language   : --lang ar")
        sys.exit(0)

    clean_text, raw_text, hallucinated, total = clean_ocr(ocr_json, args.threshold)

    if args.raw or args.threshold == 0.0:
        print("\n🔬 Raw output (word + confidence):\n")
        print(raw_text)

    print(f"\n📝 Extracted Text (threshold={args.threshold}):\n")
    print(clean_text if clean_text.strip() else "(all words filtered by threshold — try --threshold 0.0)")

    print("\n📊 Stats:")
    print(f"Total words       : {total}")
    print(f"Low-conf [UNK]    : {hallucinated}")
    if total > 0:
        print(f"Hallucination rate: {hallucinated/total:.2%}")

    if args.output:
        out = Path(args.output)
        with open(out, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"\n💾 Saved to: {out}")