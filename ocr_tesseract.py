#!/usr/bin/env python3
"""
Tesseract OCR - Extract text from images (JPG/PNG) and PDFs
Usage: python tesseract_ocr.py <file_path> [--lang eng] [--threshold 0.0] [--raw]
Install: pip install pytesseract pdf2image pillow
System:  sudo apt install tesseract-ocr        (base)
         sudo apt install tesseract-ocr-fra     (French)
         sudo apt install tesseract-ocr-ara     (Arabic)
         tesseract --list-langs                 (see installed)
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import pytesseract
    from PIL import Image
except ImportError:
    os.system(f"{sys.executable} -m pip install pytesseract pillow")
    import pytesseract
    from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    os.system(f"{sys.executable} -m pip install pdf2image")
    from pdf2image import convert_from_path

DEFAULT_THRESHOLD = 0.75


# ---------------- OCR ---------------- #
def run_ocr_on_image(image: Image.Image, lang: str) -> list:
    """Run Tesseract on a PIL Image. Returns list of {word, confidence}."""
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

    words = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = int(data["conf"][i])
        if conf == -1:
            continue
        words.append({"word": word, "confidence": round(conf / 100.0, 4)})

    return words


def run_ocr(file_path: str, lang: str) -> list:
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
            page_words = run_ocr_on_image(img, lang)
            print(f"{len(page_words)} words")
            all_words.extend(page_words)
    else:
        print(f"🔍 Running OCR on {path.name} (lang={lang})...")
        img = Image.open(str(path))
        all_words = run_ocr_on_image(img, lang)

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
    parser = argparse.ArgumentParser(description="Tesseract OCR text extractor")
    parser.add_argument("input", help="Image or PDF file")
    parser.add_argument("--lang", "-l", default="eng",
                        help="Tesseract language code (default: eng). "
                             "Examples: fra=French, ara=Arabic, eng+fra=mixed. "
                             "Run 'tesseract --list-langs' to see installed.")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Confidence threshold 0.0-1.0 (default: {DEFAULT_THRESHOLD}). "
                              "Use 0.0 to see ALL detected text.")
    parser.add_argument("--raw", action="store_true",
                        help="Also print raw words with confidence scores")
    parser.add_argument("--output", "-o", help="Save clean text to this file")
    args = parser.parse_args()

    # Check tesseract binary
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("❌ Tesseract binary not found.")
        print("   Ubuntu/Debian : sudo apt install tesseract-ocr")
        print("   macOS         : brew install tesseract")
        sys.exit(1)

    ocr_json = run_ocr(args.input, args.lang)

    if not ocr_json:
        print("\n⚠️  No text detected at all.")
        print("Tips:")
        print("  • Check installed langs : tesseract --list-langs")
        print("  • Arabic                : sudo apt install tesseract-ocr-ara  then --lang ara")
        print("  • French                : sudo apt install tesseract-ocr-fra  then --lang fra")
        sys.exit(0)

    clean_text, raw_text, hallucinated, total = clean_ocr(ocr_json, args.threshold)

    # Always show raw when threshold=0 or --raw flag used
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