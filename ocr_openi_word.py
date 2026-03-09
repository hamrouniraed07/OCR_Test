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

The text is in Arabic.
Do NOT translate.
Do NOT guess.
Unreadable words → [UNK]

Return ONLY valid JSON:
[
  { "word": "", "confidence": 0-1 }
]
"""

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_ocr(image_path):

    base64_image = encode_image(image_path)

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


if __name__ == "__main__":

    file_path = sys.argv[1]

    print("🔍 Running OpenAI OCR...\n")

    raw = run_ocr(file_path)

    print("📝 Result:\n")
    print(raw)