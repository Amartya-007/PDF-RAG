from __future__ import annotations

from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.phrase_index import PhraseIndex


def test_heading_index_ranks_exact_before_prefix_and_token_overlap() -> None:
    index = HeadingIndex()
    index.index("exact", "Transaction States")
    index.index("prefix", "Transaction States and Recovery")
    index.index("overlap", "Recovery Transaction Log")
    index.index("unrelated", "Buffer Management")

    assert index.search("Transaction States")[:3] == ["exact", "prefix", "overlap"]


def test_heading_index_remove_and_rebuild_replace_contents() -> None:
    index = HeadingIndex()
    index.index("old", "Obsolete Section")
    index.remove("old")
    assert index.search("Obsolete Section") == []

    index.rebuild([("new", "New Heading")])

    assert index.search("Obsolete Section") == []
    assert index.search("New Heading") == ["new"]


def test_phrase_index_finds_exact_phrase_from_title_and_first_sentence() -> None:
    index = PhraseIndex()
    index.index(
        "node1",
        "Annual Leave Policy",
        "Employees may carry forward earned leave. Later text is ignored for phrase extraction.",
    )

    assert index.search('"annual leave"') == [("node1", 1.0)]
    assert index.search("carry forward earned") == [("node1", 1.0)]


def test_phrase_index_remove_and_rebuild_replace_contents() -> None:
    index = PhraseIndex()
    index.index("old", "Old Annual Leave", "Old phrase appears here.")
    index.remove("old")
    assert index.search("annual leave") == []

    index.rebuild([("new", "New Travel Policy", "Travel claims require receipts.")])

    assert index.search("annual leave") == []
    assert index.search("travel policy") == [("new", 1.0)]
