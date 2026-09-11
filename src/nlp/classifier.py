"""
Zero-shot document classification using HuggingFace BART.
Classifies documents into predefined categories without training data.
"""

from typing import List, Dict
from transformers import pipeline


DOCUMENT_LABELS = [
    "Invoice",
    "Resume or CV",
    "Research Paper",
    "Legal Contract",
    "News Article",
    "Medical Report",
    "Financial Report",
    "General Document",
]


class DocumentClassifier:
    """
    Classifies documents using zero-shot classification.
    Uses facebook/bart-large-mnli — no training data required.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", labels: List[str] = None):
        self.labels = labels or DOCUMENT_LABELS
        print(f"Loading classifier: {model_name}...")
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # CPU
        )
        print("✅ Classifier ready")

    def classify(self, text: str, top_k: int = 3) -> Dict:
        """
        Classify document text into categories.

        Args:
            text: Document text (can be truncated to first 1000 chars).
            top_k: Number of top predictions to return.

        Returns:
            dict with predicted_label, confidence, all_scores
        """
        # Use first 1000 chars for speed — enough context for classification
        text_snippet = text[:1000].strip()

        if not text_snippet:
            return {
                "predicted_label": "Unknown",
                "confidence": 0.0,
                "all_scores": {},
            }

        result = self.pipeline(
            text_snippet,
            candidate_labels=self.labels,
            multi_label=False,
        )

        all_scores = dict(zip(result["labels"], result["scores"]))
        top_labels = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return {
            "predicted_label": result["labels"][0],
            "confidence": round(result["scores"][0], 4),
            "all_scores": all_scores,
            "top_k": top_labels,
        }
