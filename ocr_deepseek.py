import os
import sys
import json
import base64
from PIL import Image
import io
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

OCR_PROMPT = """
You are a STRICT OCR engine.

Rules:
- The text language is Arabic.
- Transcribe the text in Arabic script only.
- DO NOT translate.
- DO NOT change the language.
- Copy exactly what you see.
- If a word is unclear → write [UNK]
- Keep original spelling.

Return ONLY valid JSON:
[
  { "word": "", "confidence": 0-1 }
]
"""

CONF_THRESHOLD = 0.75


# ---------- IMAGE → BASE64 ---------- #
def encode_image(path, max_size=1024, quality=85):
    img = Image.open(path)

    # Convert to RGB (important for PNG with alpha)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize (keep aspect ratio)
    img.thumbnail((max_size, max_size))

    # Compress to JPEG in memory
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------- OCR ---------- #
def run_ocr(image_path):

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role": "user",
                "content": f"{OCR_PROMPT}\n\nImage data: data:image/png;base64,{base64_image}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ---------- CLEANING ---------- #
def clean_ocr(ocr_json):

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


# ---------- MAIN ---------- #
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    print("🔍 Running DeepSeek OCR...\n")

    raw = run_ocr(file_path)

    try:
        ocr_json = json.loads(raw)
    except json.JSONDecodeError:
        print("❌ Invalid JSON returned:\n")
        print(raw)
        sys.exit(1)

    clean_text, hallucinated, total = clean_ocr(ocr_json)

    print("📝 Extracted Text:\n")
    print(clean_text)

    print("\n📊 Stats:")
    print(f"Total words: {total}")
    print(f"Hallucinated words: {hallucinated}")
    print(f"Hallucination rate: {hallucinated/total:.2%}")