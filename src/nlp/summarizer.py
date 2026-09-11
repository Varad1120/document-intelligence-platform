"""
Text summarization using HuggingFace T5/DistilBART.
Bypasses the transformers pipeline() task registry entirely.
Uses AutoModelForSeq2SeqLM directly — works with all transformers versions.
"""

from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class DocumentSummarizer:
    """
    Generates concise summaries using T5-small.
    Loads model directly (no pipeline) — avoids task registry issues.
    T5 summarization: prepend 'summarize: ' to input text.
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        max_length: int = 150,
        min_length: int = 40,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length

        print(f"Loading summarizer model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
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

        text_input = text[:3000].strip()
        word_count = len(text_input.split())

        # Return as-is for very short texts
        if word_count < 30:
            return {
                "summary": text_input,
                "original_length": word_count,
                "summary_length": word_count,
                "compression_ratio": 1.0,
                "note": "Text too short — returned as-is.",
            }

        # T5 requires "summarize: " prefix
        if self.model_name.startswith("t5"):
            prefixed = "summarize: " + text_input
        else:
            prefixed = text_input

        # Tokenize
        inputs = self.tokenizer(
            prefixed,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        max_len = min(self.max_length, max(40, word_count // 3))
        min_len = min(self.min_length, max_len - 10)

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_words = len(summary.split())
        compression = round(summary_words / max(word_count, 1), 3)

        return {
            "summary": summary,
            "original_length": word_count,
            "summary_length": summary_words,
            "compression_ratio": compression,
        }
