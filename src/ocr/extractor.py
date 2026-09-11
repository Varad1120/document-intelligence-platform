"""
OCR and text extraction from PDFs and scanned images.
Supports digital PDFs (direct text) and scanned docs (pytesseract OCR).
"""

import io
from pathlib import Path
from typing import List, Dict, Optional

import fitz                   # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


class DocumentExtractor:
    """
    Extracts text from PDFs, images, and scanned documents.

    Strategy:
    - For digital PDFs: extract text directly using PyMuPDF
    - For scanned PDFs / images: run pytesseract OCR
    """

    def __init__(self, lang: str = "eng", dpi: int = 300, min_text_length: int = 50):
        self.lang = lang
        self.dpi = dpi
        self.min_text_length = min_text_length

    def extract(self, file_path: str) -> Dict:
        """
        Main entry point. Detects file type and extracts text.

        Returns:
            dict with keys: text, pages, method, word_count, char_count
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            return self._extract_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> Dict:
        """Extract text from in-memory file bytes."""
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf_bytes(file_bytes)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            image = Image.open(io.BytesIO(file_bytes))
            return self._ocr_image(image, source=filename)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # ── PDF Extraction ──────────────────────────────────────────────────────────

    def _extract_pdf(self, path: str) -> Dict:
        with open(path, "rb") as f:
            return self._extract_pdf_bytes(f.read())

    def _extract_pdf_bytes(self, pdf_bytes: bytes) -> Dict:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            pages_text.append({"page": page_num + 1, "text": text})

        full_text = "\n\n".join(p["text"] for p in pages_text if p["text"])

        # If digital text is too short, fall back to OCR
        if len(full_text.strip()) < self.min_text_length:
            return self._ocr_pdf_bytes(pdf_bytes, doc)

        return self._build_result(full_text, pages_text, method="digital_pdf", n_pages=len(doc))

    def _ocr_pdf_bytes(self, pdf_bytes: bytes, doc: fitz.Document) -> Dict:
        """OCR each page of a scanned PDF."""
        pages_text = []

        for page_num, page in enumerate(doc):
            # Render page to image
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            text = self._run_tesseract(image)
            pages_text.append({"page": page_num + 1, "text": text})

        full_text = "\n\n".join(p["text"] for p in pages_text if p["text"])
        return self._build_result(full_text, pages_text, method="ocr_pdf", n_pages=len(doc))

    # ── Image Extraction ────────────────────────────────────────────────────────

    def _extract_image(self, path: str) -> Dict:
        image = Image.open(path)
        return self._ocr_image(image, source=path)

    def _ocr_image(self, image: Image.Image, source: str = "") -> Dict:
        text = self._run_tesseract(image)
        page = [{"page": 1, "text": text}]
        return self._build_result(text, page, method="ocr_image", n_pages=1)

    # ── Tesseract OCR ───────────────────────────────────────────────────────────

    def _run_tesseract(self, image: Image.Image) -> str:
        """Preprocess image and run tesseract OCR."""
        # Convert to RGB if needed
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Preprocessing: enhance contrast for better OCR
        image = image.convert("L")                          # Grayscale
        image = ImageEnhance.Contrast(image).enhance(2.0)  # Boost contrast
        image = image.filter(ImageFilter.SHARPEN)           # Sharpen

        config = f"--oem 3 --psm 6 -l {self.lang}"
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _build_result(self, full_text: str, pages: List[Dict], method: str, n_pages: int) -> Dict:
        words = full_text.split()
        return {
            "text": full_text,
            "pages": pages,
            "method": method,
            "n_pages": n_pages,
            "word_count": len(words),
            "char_count": len(full_text),
        }
