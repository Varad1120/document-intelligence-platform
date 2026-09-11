"""
Named Entity Recognition (NER) using spaCy.
Extracts people, organizations, dates, locations, money, etc.
"""

from typing import Dict, List
from collections import defaultdict

import spacy


ENTITY_ICONS = {
    "PERSON":   "👤",
    "ORG":      "🏢",
    "GPE":      "📍",
    "LOC":      "🗺️",
    "DATE":     "📅",
    "TIME":     "⏰",
    "MONEY":    "💰",
    "PERCENT":  "📊",
    "EMAIL":    "📧",
    "PHONE":    "📞",
    "PRODUCT":  "📦",
    "EVENT":    "🎯",
    "LAW":      "⚖️",
    "MISC":     "📌",
}


class EntityExtractor:
    """
    Extracts named entities from document text using spaCy.
    Model: en_core_web_sm (fast, good accuracy for common entities)
    """

    def __init__(self, model: str = "en_core_web_sm"):
        print(f"Loading spaCy model: {model}...")
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Model {model} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            self.nlp = spacy.load(model)
        print("✅ NER model ready")

    def extract(self, text: str) -> Dict:
        """
        Extract all named entities from text.

        Returns:
            dict with entities grouped by type, and flat list of all entities
        """
        # Process in chunks to handle long docs
        max_length = 100_000
        text = text[:max_length]

        doc = self.nlp(text)

        grouped = defaultdict(list)
        all_entities = []

        seen = set()
        for ent in doc.ents:
            key = (ent.label_, ent.text.strip())
            if key in seen:
                continue
            seen.add(key)

            entity = {
                "text":  ent.text.strip(),
                "label": ent.label_,
                "desc":  spacy.explain(ent.label_) or ent.label_,
                "icon":  ENTITY_ICONS.get(ent.label_, "📌"),
            }
            grouped[ent.label_].append(entity)
            all_entities.append(entity)

        return {
            "entities": all_entities,
            "grouped":  dict(grouped),
            "counts":   {label: len(ents) for label, ents in grouped.items()},
            "total":    len(all_entities),
        }

    def get_summary_stats(self, entities: Dict) -> List[Dict]:
        """Return entity counts as a list for display."""
        return [
            {
                "Type":  f"{ENTITY_ICONS.get(label, '📌')} {label}",
                "Description": spacy.explain(label) or label,
                "Count": count,
            }
            for label, count in sorted(
                entities["counts"].items(), key=lambda x: x[1], reverse=True
            )
        ]
