from __future__ import annotations
import re
from backend.app.domain.enums import QueryType

class QueryClassifier:
    """Classifies queries using a clean map-based routing approach."""

    # Map patterns to their target Enums
    # The order here defines your priority (First Match Wins)
    _ROUTING_TABLE = {
        re.compile(
            r"\b(name|email|phone|contact|address|date|location|"
            r"college|university|institute|school|academy|campus|"
            r"department|dept|cgpa|gpa|percentage)\b",
            re.I,
        ): QueryType.FAST_FACT,
        re.compile(r"\b(what is|what are|define|explain|describe|details? about)\b", re.I): QueryType.TOPIC,
        re.compile(r"\b(compare|versus|vs\.?)\b", re.I): QueryType.COMPARISON,
        re.compile(r"\b(summarize|summary|overview)\b", re.I): QueryType.SUMMARY,
        re.compile(r"\b(table|highest|lowest|total|amount|number)\b", re.I): QueryType.NUMERIC_OR_TABLE,
        re.compile(r"\b(id|order|clause|section|code)\b", re.I): QueryType.EXACT_IDENTIFIER,
    }

    def classify(self, question: str) -> QueryType:
        # Standardize input
        norm = question.lower().replace("collage", "college").replace("transection", "transaction")
        
        # O(N) loop over a fixed number of patterns
        for pattern, q_type in self._ROUTING_TABLE.items():
            if pattern.search(norm):
                return q_type
        
        # Fallback logic
        return QueryType.FOLLOW_UP_OR_SHORT if len(question.split()) <= 4 else QueryType.DIRECT_FACTUAL