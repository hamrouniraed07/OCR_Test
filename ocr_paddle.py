#!/usr/bin/env python3
"""
PaddleOCR - Extract text from images (JPG/PNG) and PDFs
Usage: python paddle_ocr.py <file_path>
Install: pip install paddlepaddle paddleocr pdf2image
"""

import os
import sys
import json
from pathlib import Path

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Installing paddleocr...")
    os.system(f"{sys.executable} -m pip install paddleocr paddlepaddle")
    from paddleocr import PaddleOCR

try:
    from pdf2image import convert_from_path
except ImportError:
    os.system(f"{sys.executable} -m pip install pdf2image")
    from pdf2image import convert_from_path

import tempfile
from dotenv import load_dotenv

load_dotenv()

CONF_THRESHOLD = 0.75

# Init PaddleOCR (downloads model on first run)
# `show_log` is not supported in this paddleocr version; replace deprecated
# `use_angle_cls` with `use_textline_orientation`.
ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")


# ---------------- OCR ---------------- #
def run_ocr_on_image(image_path: str) -> list:
    """Run PaddleOCR on a single image file. Returns list of {word, confidence}."""
    # Prefer `predict` (newer paddleocr); fall back to `ocr` if needed.
    try:
        result = ocr_engine.predict(image_path)
    except TypeError:
        try:
            result = ocr_engine.ocr(image_path, cls=True)
        except Exception:
            return []
    except Exception:
        return []

    words = []
    if not result:
        return words

    # Normalize different possible result shapes across paddleocr versions
    # Common shapes:
    #  - old: result[0] -> list of [box, (text, score)]
    #  - new: result -> list of [box, (text, score)] or list of tuples
    lines = None
    if isinstance(result, list):
        if result and isinstance(result[0], list) and result[0] and isinstance(result[0][0], (list, tuple)):
            # result is nested: [ [lines...] ]
            lines = result[0]
        else:
            lines = result

    if not lines:
        return words

    for line in lines:
        # line expected to be [box, rec] where rec is (text, score)
        text = None
        conf = None
        if isinstance(line, (list, tuple)) and len(line) >= 2:
            rec = line[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text = rec[0]
                conf = rec[1]
            elif isinstance(rec, str):
                text = rec
        # If still no text, try a fallback when line itself may be (text, score)
        if text is None:
            if isinstance(line, (list, tuple)) and len(line) >= 2 and isinstance(line[0], str):
                text = line[0]
                conf = line[1]

        if not text:
            continue

        # Split multi-word lines into individual words
        try:
            conf_val = float(conf) if conf is not None else 1.0
        except Exception:
            conf_val = 1.0
        for word in str(text).split():
            words.append({"word": word, "confidence": round(float(conf_val), 4)})

    return words


def run_ocr(file_path: str) -> list:
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    all_words = []

    if path.suffix.lower() == ".pdf":
        print("📄 Converting PDF pages to images...")
        images = convert_from_path(str(path), dpi=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(images):
                img_path = os.path.join(tmpdir, f"page_{i+1}.png")
                img.save(img_path, "PNG")
                print(f"  🔍 OCR on page {i+1}/{len(images)}...")
                all_words.extend(run_ocr_on_image(img_path))
    else:
        print(f"🔍 Running OCR on {path.name}...")
        all_words = run_ocr_on_image(str(path))

    return all_words


# ---------------- CLEANING ---------------- #
def clean_ocr(ocr_json: list):
    clean_words = []
    hallucinated = 0
    total = 0

    for w in ocr_json:
        total += 1
        if w["confidence"] < CONF_THRESHOLD:
            clean_words.append("[UNK]")
            hallucinated += 1
        else:
            clean_words.append(w["word"])

    return " ".join(clean_words), hallucinated, total


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python paddle_ocr.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    ocr_json = run_ocr(file_path)

    if not ocr_json:
        print("⚠️  No text detected.")
        sys.exit(0)

    clean_text, hallucinated, total = clean_ocr(ocr_json)

    print("\n📝 Extracted Text:\n")
    print(clean_text)

    print("\n📊 Stats:")
    print(f"Total words : {total}")
    print(f"Low-conf    : {hallucinated}")
    print(f"Hallucination rate: {hallucinated/total:.2%}")