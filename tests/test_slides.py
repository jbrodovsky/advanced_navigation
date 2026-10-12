"""Checks that the lecture decks in slides/decks/ stay mapped to the book.

Each deck lists the book chapters it covers in its front matter
(`book-sources`). Every teaching chapter of the book must belong to exactly one
deck, so that a change to a chapter always has a deck to review.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
DECKS = sorted((ROOT / "slides" / "decks").glob("[0-9][0-9]-*.qmd"))

# Book entries with no deck: the front matter, the bibliography, and the
# optional appendix.
NOT_TAUGHT = {"index.qmd", "references.qmd"}
NOT_TAUGHT_PARTS = {"Technical Reference (Optional)"}


def front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path.name} has no YAML front matter"
    return yaml.safe_load(text.split("---\n", 2)[1])


def teaching_chapters() -> set[str]:
    """Repository-relative paths of every book chapter that needs a deck."""
    config = yaml.safe_load((BOOK / "_quarto.yml").read_text(encoding="utf-8"))
    chapters = set()
    for entry in config["book"]["chapters"]:
        if isinstance(entry, str):
            files = [entry]
        elif entry["part"] in NOT_TAUGHT_PARTS:
            files = []
        else:
            files = entry["chapters"]
        chapters.update(f"book/{f}" for f in files if f not in NOT_TAUGHT)
    return chapters


def test_there_is_one_deck_per_part():
    assert [int(deck.name[:2]) for deck in DECKS] == list(range(1, 9))


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_deck_front_matter(deck):
    meta = front_matter(deck)
    assert meta["part"] == int(deck.name[:2]), "`part` must match the file number"
    for key in ("title", "footer", "book-sources"):
        assert meta.get(key), f"missing `{key}`"
    for source in meta["book-sources"]:
        assert (ROOT / source).is_file(), f"book-sources entry {source} does not exist"


def test_every_teaching_chapter_is_in_exactly_one_deck():
    owners: dict[str, list[str]] = {}
    for deck in DECKS:
        for source in front_matter(deck)["book-sources"]:
            owners.setdefault(source, []).append(deck.name)
    missing = sorted(teaching_chapters() - owners.keys())
    duplicated = {source: decks for source, decks in owners.items() if len(decks) > 1}
    assert not missing, f"chapters with no deck: {missing}"
    assert not duplicated, f"chapters in more than one deck: {duplicated}"
