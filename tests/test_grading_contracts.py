"""Checks that the project rubrics follow the contract-grading policy.

The policy (book/chapters/00-intro/intro.qmd, "Grading and Project Submission")
says every rubric row is worth 20 points, rated Unacceptable (0), Marginal
(15), Acceptable (17), or Excellent (20), so that a project rated Acceptable on
every row earns 85%. The ratings are whole numbers because Canvas rubrics need
them. Each handout's rubric must use exactly those columns, number its rows
1..N, and state N and the point total in its introduction. The Project 1 part
handouts (A, B, C) repeat their rows of the Project 1 rubric, which must match
the full rubric word for word. The syllabus weights must add up to 100%.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"

HANDOUTS = {
    1: BOOK / "assignments" / "m1-drone-pipeline.qmd",
    2: BOOK / "assignments" / "m2-victoria-park.qmd",
    3: BOOK / "assignments" / "m3-ins-gnss.qmd",
}
COLUMNS = ["Category", "Unacceptable (0)", "Marginal (15)", "Acceptable (17)", "Excellent (20)"]
POINTS_PER_ROW = 20
PART_ROWS = {"a": [2, 3, 4], "b": [5, 6, 7], "c": [8, 9, 10, 11, 12]}


def section_text(path: Path, anchor: str) -> str:
    text = path.read_text(encoding="utf-8")
    assert anchor in text, f"{path.name} has no {anchor} section"
    return text.split(anchor, 1)[1].split("\n## ", 1)[0]


def rubric_section(project: int) -> str:
    return section_text(HANDOUTS[project], f"{{#sec-project{project}-grading-rubric}}")


def table_rows(text: str) -> dict[int, str]:
    """Map each numbered rubric row to its grid-table lines."""
    blocks = re.split(r"^\+[-=+]+\+$", text, flags=re.MULTILINE)
    rows = {}
    for block in blocks:
        match = re.match(r"\s*\| \*\*(\d+)\. ", block)
        if match:
            rows[int(match.group(1))] = block.strip()
    return rows


@pytest.mark.parametrize("project", sorted(HANDOUTS))
def test_rubric_columns_are_the_four_ratings(project: int) -> None:
    section = rubric_section(project)
    header = re.search(r"^\| Category .*\|$", section, flags=re.MULTILINE)
    assert header, f"Project {project} rubric has no Category header row"
    columns = [cell.strip() for cell in header.group(0).strip("|").split("|")]
    assert columns == COLUMNS


@pytest.mark.parametrize("project", sorted(HANDOUTS))
def test_rows_are_numbered_and_counted(project: int) -> None:
    section = rubric_section(project)
    numbers = [int(n) for n in re.findall(r"^\| \*\*(\d+)\. ", section, flags=re.MULTILINE)]
    assert numbers, f"Project {project} rubric has no numbered rows"
    assert numbers == list(range(1, len(numbers) + 1)), numbers
    count = len(numbers)
    assert f"Each of the {count} rows" in section
    assert f"({POINTS_PER_ROW * count} in all)" in section


def test_syllabus_weights_add_up() -> None:
    text = (BOOK / "chapters" / "00-intro" / "syllabus.qmd").read_text(encoding="utf-8")
    section = text.split("{#sec-syllabus-grading-policy}", 1)[1].split("\n## ", 1)[0]
    weights = [int(w) for w in re.findall(r"^- Project \d: .* -- (\d+)%$", section, flags=re.MULTILINE)]
    assert len(weights) == len(HANDOUTS)
    assert sum(weights) == 100
    for project, weight in zip(sorted(HANDOUTS), weights):
        assert f"counts for {weight}% of the course grade" in rubric_section(project)


@pytest.mark.parametrize("part", sorted(PART_ROWS))
def test_project1_part_rubrics_repeat_the_full_rubric(part: str) -> None:
    full = table_rows(rubric_section(1))
    path = BOOK / "assignments" / f"m1-drone-pipeline-{part}.qmd"
    subset_text = section_text(path, f"{{#sec-project1{part}-grading-rubric}}")
    subset = table_rows(subset_text)
    assert sorted(subset) == PART_ROWS[part]
    for number, block in subset.items():
        assert block == full[number], f"Part {part.upper()} row {number} differs from the Project 1 rubric"
    header = re.search(r"^\| Category .*\|$", subset_text, flags=re.MULTILINE)
    assert header and [c.strip() for c in header.group(0).strip("|").split("|")] == COLUMNS
