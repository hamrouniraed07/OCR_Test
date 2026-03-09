import streamlit as st
import os
import sys
import json
import tempfile
from pathlib import Path
from PIL import Image
import time

# Import all OCR modules
try:
    from ocr_tesseract import run_ocr as tesseract_ocr, clean_ocr as tesseract_clean
except ImportError:
    tesseract_ocr = None

try:
    from ocr_paddle import run_ocr as paddle_ocr, clean_ocr as paddle_clean
except ImportError:
    paddle_ocr = None

try:
    from ocr_openai import run_ocr as openai_ocr, clean_ocr as openai_clean
except ImportError:
    openai_ocr = None

try:
    from ocr_gemini import run_ocr as gemini_ocr, clean_ocr as gemini_clean
except ImportError:
    gemini_ocr = None

try:
    from ocr_deepseek import run_ocr as deepseek_ocr, clean_ocr as deepseek_clean
except ImportError:
    deepseek_ocr = None

try:
    from ocr_openi_word import run_ocr as openi_word_ocr
except ImportError:
    openi_word_ocr = None

# Page configuration
st.set_page_config(
    page_title="OCR Model Testing Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ccc;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 OCR Model Testing Interface</h1>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
available_models = []
if tesseract_ocr:
    available_models.append("Tesseract")
if paddle_ocr:
    available_models.append("PaddleOCR")
if openai_ocr:
    available_models.append("OpenAI GPT-4o-mini")
if gemini_ocr:
    available_models.append("Gemini 3 Pro")
if deepseek_ocr:
    available_models.append("DeepSeek Reasoner")
if openi_word_ocr:
    available_models.append("OpenAI Word (Arabic)")

if not available_models:
    st.error("No OCR models available! Please check your imports and dependencies.")
    st.stop()

selected_models = st.sidebar.multiselect(
    "Select OCR Models to Test",
    available_models,
    default=available_models[:2] if len(available_models) >= 2 else available_models
)

# Tesseract language selection
if "Tesseract" in selected_models:
    tesseract_lang = st.sidebar.selectbox(
        "Tesseract Language",
        ["eng", "fra", "ara", "eng+fra", "eng+ara", "fra+ara"],
        index=0
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05
    )

# File upload
st.header("📁 Upload File")
uploaded_file = st.file_uploader(
    "Choose an image or PDF file",
    type=["jpg", "jpeg", "png", "pdf"],
    help="Supported formats: JPG, PNG, PDF"
)

if uploaded_file is not None:
    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
    with col3:
        file_type = uploaded_file.name.split('.')[-1].upper()
        st.metric("File Type", file_type)
    
    # Display image preview if it's an image
    if file_type in ["JPG", "JPEG", "PNG"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process button
    if st.button("🚀 Run OCR", type="primary") and selected_models:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type.lower()}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Create progress container
            progress_container = st.container()
            
            # Results storage
            all_results = {}
            
            for model in selected_models:
                with progress_container:
                    st.subheader(f"🔍 Processing with {model}")
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Update progress
                        progress_bar.progress(25)
                        status_text.text("Initializing OCR engine...")
                        time.sleep(0.5)
                        
                        # Run OCR based on selected model
                        progress_bar.progress(50)
                        status_text.text("Extracting text...")
                        
                        if model == "Tesseract":
                            raw_result = tesseract_ocr(tmp_file_path, tesseract_lang)
                            clean_text, raw_text, hallucinated, total = tesseract_clean(raw_result, confidence_threshold)
                            result = {
                                "raw": raw_result,
                                "clean": clean_text,
                                "raw_detailed": raw_text,
                                "stats": {
                                    "total": total,
                                    "hallucinated": hallucinated,
                                    "rate": hallucinated/total if total > 0 else 0
                                }
                            }
                        elif model == "PaddleOCR":
                            raw_result = paddle_ocr(tmp_file_path)
                            clean_text, hallucinated, total = paddle_clean(raw_result)
                            result = {
                                "raw": raw_result,
                                "clean": clean_text,
                                "stats": {
                                    "total": total,
                                    "hallucinated": hallucinated,
                                    "rate": hallucinated/total if total > 0 else 0
                                }
                            }
                        elif model == "OpenAI GPT-4o-mini":
                            raw_result = openai_ocr(tmp_file_path)
                            try:
                                ocr_json = json.loads(raw_result)
                                clean_text, hallucinated, total = openai_clean(ocr_json)
                                result = {
                                    "raw": raw_result,
                                    "clean": clean_text,
                                    "stats": {
                                        "total": total,
                                        "hallucinated": hallucinated,
                                        "rate": hallucinated/total if total > 0 else 0
                                    }
                                }
                            except json.JSONDecodeError as e:
                                st.error(f"❌ OpenAI returned invalid JSON: {str(e)}")
                                st.text_area("Raw Response", raw_result, height=200)
                                result = {
                                    "raw": raw_result,
                                    "clean": "JSON parsing failed",
                                    "stats": {
                                        "total": "Error",
                                        "hallucinated": "Error", 
                                        "rate": "Error"
                                    }
                                }
                        elif model == "Gemini 3 Pro":
                            raw_result = gemini_ocr(tmp_file_path)
                            # Gemini now returns plain text, not JSON
                            result = {
                                "raw": raw_result,
                                "clean": raw_result.strip(),
                                "stats": {
                                    "total": "N/A",
                                    "hallucinated": "N/A",
                                    "rate": "N/A"
                                }
                            }
                        elif model == "DeepSeek Reasoner":
                            raw_result = deepseek_ocr(tmp_file_path)
                            try:
                                ocr_json = json.loads(raw_result)
                                clean_text, hallucinated, total = deepseek_clean(ocr_json)
                                result = {
                                    "raw": raw_result,
                                    "clean": clean_text,
                                    "stats": {
                                        "total": total,
                                        "hallucinated": hallucinated,
                                        "rate": hallucinated/total if total > 0 else 0
                                    }
                                }
                            except json.JSONDecodeError as e:
                                st.error(f"❌ DeepSeek returned invalid JSON: {str(e)}")
                                st.text_area("Raw Response", raw_result, height=200)
                                result = {
                                    "raw": raw_result,
                                    "clean": "JSON parsing failed",
                                    "stats": {
                                        "total": "Error",
                                        "hallucinated": "Error", 
                                        "rate": "Error"
                                    }
                                }
                        elif model == "OpenAI Word (Arabic)":
                            raw_result = openi_word_ocr(tmp_file_path)
                            result = {
                                "raw": raw_result,
                                "clean": raw_result,  # This model returns raw text
                                "stats": {
                                    "total": "N/A",
                                    "hallucinated": "N/A",
                                    "rate": "N/A"
                                }
                            }
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Complete!")
                        
                        all_results[model] = result
                        
                    except Exception as e:
                        st.error(f"❌ Error with {model}: {str(e)}")
                        progress_bar.progress(0)
                        status_text.text("❌ Failed")
                    
                    st.divider()
            
            # Display results
            if all_results:
                st.header("📊 Results Comparison")
                
                # Create comparison table
                comparison_data = []
                for model, result in all_results.items():
                    stats = result["stats"]
                    comparison_data.append({
                        "Model": model,
                        "Total Words": stats["total"],
                        "Low Confidence": stats["hallucinated"],
                        "Hallucination Rate": f"{stats['rate']:.2%}" if isinstance(stats['rate'], float) else stats['rate']
                    })
                
                st.dataframe(comparison_data, use_container_width=True)
                
                # Detailed results for each model
                for model, result in all_results.items():
                    with st.expander(f"🔍 {model} - Detailed Results"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📝 Extracted Text")
                            st.text_area("Clean Text", result["clean"], height=200, key=f"clean_{model}")
                        
                        with col2:
                            st.subheader("📈 Statistics")
                            stats = result["stats"]
                            st.metric("Total Words", stats["total"])
                            st.metric("Low Confidence Words", stats["hallucinated"])
                            if isinstance(stats['rate'], float):
                                st.metric("Hallucination Rate", f"{stats['rate']:.2%}")
                            
                            if "raw_detailed" in result:
                                st.subheader("🔬 Raw Output")
                                st.text_area("Raw with Confidence", result["raw_detailed"], height=150, key=f"raw_{model}")
                        
                        # Raw JSON output
                        if st.checkbox(f"Show Raw JSON - {model}", key=f"json_{model}"):
                            st.json(result["raw"])
                
                
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

elif uploaded_file is None:
    st.info("👆 Please upload a file to begin OCR processing.")

# Instructions
with st.expander("📖 Instructions & Tips"):
    st.markdown("""
    ### How to Use:
    1. **Select Models**: Choose which OCR models to test from the sidebar
    2. **Configure Settings**: Adjust language and confidence settings for Tesseract
    3. **Upload File**: Upload an image (JPG/PNG) or PDF file
    4. **Run OCR**: Click the processing button to extract text
    5. **Compare Results**: View and compare results from different models
    
    ### Model Information:
    - **Tesseract**: Traditional OCR engine, good for structured text
    - **PaddleOCR**: Deep learning-based, supports multiple languages
    - **OpenAI GPT-4o-mini**: AI-powered OCR with context understanding
    - **Gemini 3 Pro**: Google's multimodal AI model
    - **DeepSeek Reasoner**: Advanced reasoning model for OCR
    - **OpenAI Word (Arabic)**: Specialized for Arabic text recognition
    
    ### Tips:
    - Use high-quality images for better results
    - For PDFs, each page will be processed separately
    - Adjust confidence threshold to filter low-confidence results
    - Compare results across models to find the best for your use case
    """)

# Footer
st.markdown("---")
st.markdown("🔍 OCR Model Testing Interface | Built with Streamlit")
