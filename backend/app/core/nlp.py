"""Shared spaCy pipeline singleton.

Both the generation layer (name/institution extraction) and the retrieval
layer (person-name ranking signal) need lightweight NER. Loading the model
in one place means it's only loaded into memory once per process.
"""
from __future__ import annotations

import spacy

nlp = spacy.load("en_core_web_sm")
