"""
Text summarization using HuggingFace DistilBART.
Generates abstractive summaries of document text.
Uses text2text-generation task (compatible with transformers >= 4.40).
"""

from typing import Dict
from transformers import pipeline


class DocumentSummarizer:
    """
    Generates concise summaries of document text using DistilBART.
    Model: sshleifer/distilbart-cnn-12-6 (faster, smaller than bart-large-cnn)
    """

    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        max_length: int = 150,
        min_length: int = 40,
    ):
        self.max_length = max_length
        self.min_length = min_length
        print(f"Loading summarizer: {model_name}...")
        # Use text2text-generation — compatible with all modern transformers versions
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=-1,
        )
        print("✅ Summarizer ready")

    def summarize(self, text: str) -> Dict:
        """
        Generate a summary of the input text.

        Args:
            text: Document text to summarize.

        Returns:
            dict with summary, original_length, summary_length, compression_ratio
        """
        if not text or not text.strip():
            return {
                "summary": "No text available to summarize.",
                "original_length": 0,
                "summary_length": 0,
                "compression_ratio": 0.0,
            }

        # DistilBART max input is ~1024 tokens; ~4 chars/token → ~4000 chars
        text_input = text[:4000].strip()
        word_count = len(text_input.split())

        # Skip for very short texts
        if word_count < 30:
            return {
                "summary": text_input,
                "original_length": word_count,
                "summary_length": word_count,
                "compression_ratio": 1.0,
                "note": "Text too short — returned as-is.",
            }

        max_len = min(self.max_length, max(40, word_count // 3))
        min_len = min(self.min_length, max_len - 10)

        result = self.pipeline(
            text_input,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )

        # text2text-generation returns 'generated_text' (not 'summary_text')
        summary = result[0]["generated_text"]
        summary_words = len(summary.split())
        compression = round(summary_words / max(word_count, 1), 3)

        return {
            "summary": summary,
            "original_length": word_count,
            "summary_length": summary_words,
            "compression_ratio": compression,
        }

