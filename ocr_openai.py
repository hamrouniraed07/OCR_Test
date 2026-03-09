import os
import sys
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OCR_PROMPT = """
You are a STRICT OCR engine.

The text may be in Arabic, English, or French.
Transcribe only visible text in the original language.
Do NOT translate.
Do NOT guess.
Unreadable words → [UNK]
- Return ONLY the extracted text, nothing else
"""

CONF_THRESHOLD = 0.75


# ---------------- IMAGE → BASE64 ---------------- #
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------- OCR ---------------- #
def run_ocr(file_path):

    print(f"📤 Sending {file_path} to OpenAI...")

    base64_image = encode_image(file_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


# ---------------- CLEANING ---------------- #
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


# ---------------- MAIN ---------------- #
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python ocr_openai.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    print("🔍 Running OpenAI OCR...")

    raw = run_ocr(file_path)

    # Convert → JSON
    try:
        ocr_json = json.loads(raw)
    except json.JSONDecodeError:
        print("❌ OpenAI did not return valid JSON:\n")
        print(raw)
        sys.exit(1)

    clean_text, hallucinated, total = clean_ocr(ocr_json)

    print("\n📝 Extracted Text:\n")
    print(clean_text)

    print("\n📊 Stats:")
    print(f"Total words: {total}")
    print(f"Hallucinated words: {hallucinated}")
    print(f"Hallucination rate: {hallucinated/total:.2%}")