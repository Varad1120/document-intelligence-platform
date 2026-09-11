# 📄 AI-Based Document Intelligence Platform

> An end-to-end **Document Intelligence System** that extracts, classifies, summarizes, and semantically searches documents using NLP, OCR, and Transformer models.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Project Overview

This platform provides intelligent document processing capabilities:

| Feature | Description |
|---------|-------------|
| 📤 **Document Ingestion** | Upload PDFs, images, and scanned documents |
| 🔍 **OCR Extraction** | Extract text from scanned docs using pytesseract |
| 🏷️ **Document Classification** | Auto-classify into Invoice, Resume, Report, Legal, etc. |
| 🧠 **Named Entity Recognition** | Extract people, organizations, dates, locations |
| ✍️ **Text Summarization** | Generate concise summaries using BART/T5 |
| 🔎 **Semantic Search** | Find similar documents using FAISS + sentence embeddings |
| 📊 **Analytics Dashboard** | Document insights and processing statistics |

---

## 🏗️ Project Structure

```
document-intelligence-platform/
├── src/
│   ├── ocr/
│   │   └── extractor.py        # PDF & image text extraction
│   ├── nlp/
│   │   ├── classifier.py       # Document type classification
│   │   ├── ner.py              # Named entity recognition
│   │   └── summarizer.py       # Text summarization
│   ├── search/
│   │   └── semantic_search.py  # FAISS-based semantic search
│   └── utils/
│       ├── preprocessor.py     # Text cleaning & preprocessing
│       └── file_handler.py     # File I/O utilities
├── app/
│   └── streamlit_app.py        # Main Streamlit dashboard
├── data/
│   ├── uploads/                # User uploaded documents
│   ├── processed/              # Extracted & processed text
│   └── sample_docs/            # Sample documents for demo
├── configs/
│   └── config.yaml             # App configuration
├── notebooks/
│   └── demo.ipynb              # Jupyter demo notebook
├── tests/
│   └── test_pipeline.py        # Unit tests
├── requirements.txt
└── README.md
```

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **OCR** | pytesseract, pdf2image, PyMuPDF |
| **NLP Models** | HuggingFace Transformers (BERT, BART, T5) |
| **NER** | spaCy (en_core_web_sm) |
| **Classification** | Zero-shot classification (facebook/bart-large-mnli) |
| **Summarization** | facebook/bart-large-cnn |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Semantic Search** | FAISS |
| **Dashboard** | Streamlit |
| **File Handling** | PyMuPDF, Pillow, python-docx |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Varad1120/document-intelligence-platform.git
cd document-intelligence-platform
```

### 2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate.fish   # fish shell
# OR
source venv/bin/activate        # bash/zsh
pip install -r requirements.txt
```

### 3. Install system dependencies (macOS)
```bash
brew install tesseract poppler
```

### 4. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## ✨ Features in Detail

### 🔍 OCR Text Extraction
- Supports **PDF**, **PNG**, **JPG**, **TIFF** formats
- Handles both **digital PDFs** (direct text) and **scanned images** (OCR)
- Multi-page document processing

### 🏷️ Document Classification
- Classifies into: **Invoice**, **Resume/CV**, **Research Paper**, **Legal Contract**, **News Article**, **Medical Report**, **Financial Report**
- Uses zero-shot classification — no training data needed

### 🧠 Named Entity Recognition
- Extracts: **PERSON**, **ORG**, **DATE**, **LOCATION**, **MONEY**, **EMAIL**, **PHONE**
- Powered by spaCy's pretrained English model

### ✍️ Summarization
- Generates 3-5 sentence abstractive summaries
- Uses BART-large-CNN fine-tuned on CNN/DailyMail dataset

### 🔎 Semantic Search
- Convert documents to dense vector embeddings
- Index with FAISS for millisecond-speed similarity search
- Find relevant documents by natural language query

---

## 🗺️ Roadmap

- [x] Project structure & repository setup
- [x] OCR text extraction pipeline
- [x] Document classification (zero-shot)
- [x] Named Entity Recognition
- [x] Text summarization
- [x] Semantic search with FAISS
- [x] Streamlit dashboard with landing page
- [ ] REST API with FastAPI
- [ ] Docker containerization
- [ ] Support for DOCX files

---

## 👨‍💻 Author

**Varad Sachin Kale** · [GitHub @Varad1120](https://github.com/Varad1120)
📧 varadk1120@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
