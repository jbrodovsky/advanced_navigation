"""Checks that the project rubrics follow the contract-grading policy.

The policy (book/chapters/00-intro/intro.qmd, "Grading and Project Submission")
says every rubric row is worth 10 points, rated Unacceptable (0), Marginal
(7.5), Acceptable (8.5), or Excellent (10), so that a project rated Acceptable
on every row earns 85%. Each handout's rubric must use exactly those columns,
number its rows 1..N, and state N and the point total in its introduction. The
syllabus weights must add up to 100%.
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
COLUMNS = ["Category", "Unacceptable (0)", "Marginal (7.5)", "Acceptable (8.5)", "Excellent (10)"]


def rubric_section(project: int) -> str:
    text = HANDOUTS[project].read_text(encoding="utf-8")
    anchor = f"{{#sec-project{project}-grading-rubric}}"
    assert anchor in text, f"Project {project} handout has no {anchor} section"
    return text.split(anchor, 1)[1].split("\n## ", 1)[0]


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
    assert f"({10 * count} in all)" in section


def test_syllabus_weights_add_up() -> None:
    text = (BOOK / "chapters" / "00-intro" / "syllabus.qmd").read_text(encoding="utf-8")
    section = text.split("{#sec-syllabus-grading-policy}", 1)[1].split("\n## ", 1)[0]
    weights = [int(w) for w in re.findall(r"^- Project \d: .* -- (\d+)%$", section, flags=re.MULTILINE)]
    assert len(weights) == len(HANDOUTS)
    assert sum(weights) == 100
    for project, weight in zip(sorted(HANDOUTS), weights):
        assert f"counts for {weight}% of the course grade" in rubric_section(project)
