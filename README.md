# OCR Model Testing Project

This project compares multiple OCR backends on images and PDFs using a Streamlit interface.

## Included OCR Backends

- Tesseract
- PaddleOCR
- OpenAI (GPT-4o-mini)
- Gemini
- DeepSeek
- OpenAI Word (Arabic)

## Project Files

- `streamlit_ocr_app.py`: main web UI to test and compare OCR models.
- `ocr_tesseract.py`: Tesseract OCR implementation.
- `ocr_paddle.py`: PaddleOCR implementation.
- `ocr_openai.py`: OpenAI OCR implementation.
- `ocr_gemini.py`: Gemini OCR implementation.
- `ocr_deepseek.py`: DeepSeek OCR implementation.
- `ocr_openi_word.py`: OpenAI Arabic-word focused OCR script.

## Requirements

- Python 3.10+
- Linux/macOS/Windows
- System package for Tesseract if you plan to use Tesseract model

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root (only keys you need):

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
```

## Run The Streamlit App

```bash
streamlit run streamlit_ocr_app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

## Run Individual OCR Scripts

```bash
python ocr_tesseract.py <path_to_image_or_pdf>
python ocr_paddle.py <path_to_image_or_pdf>
python ocr_openai.py <path_to_image>
python ocr_gemini.py <path_to_image>
python ocr_deepseek.py <path_to_image>
python ocr_openi_word.py <path_to_image>
```

## Notes

- Some models may require first-run downloads (for example PaddleOCR models).
- OCR quality is strongly affected by image quality and language.
- For PDFs, conversion uses `pdf2image` and processes pages as images.

## GitHub

Your `.gitignore` already excludes:

- `venv/` and other virtual env folders
- `.env` and secret files
- Python cache/build artifacts
- editor/system files
