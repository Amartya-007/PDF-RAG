from __future__ import annotations

import re


def classify_query(question: str) -> str:
    lowered = question.lower()
    if re.search(r"\b(compare|difference|versus|vs\.?)\b", lowered):
        return "comparison"
    if re.search(r"\b(summarize|summary|overview)\b", lowered):
        return "summary"
    if re.search(r"\b(table|highest|lowest|total|amount|number|date|when)\b", lowered):
        return "numeric_or_table"
    if re.search(r"\b(id|order|clause|section|code|no\.)\b", lowered):
        return "exact_identifier"
    if len(question.split()) <= 4:
        return "follow_up_or_short"
    return "direct_factual"
