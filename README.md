# Advanced Robot Navigation

Course notes, project handouts, and Python starter code for **RBE 595: Advanced
Robot Navigation** (WPI). The course takes students with some robotics
background to the point where they can design, implement, and evaluate the
navigation system of an autonomous vehicle, from perception-derived
measurements through filtering and SLAM to INS/GNSS integration.

© 2026 James Brodovsky. All rights reserved. This repository is private and its
contents are proprietary; do not redistribute.

## Layout

| Path | Contents |
|---|---|
| `book/` | Quarto book: the course reference notes, syllabus, and project handouts (`book/assignments/`) |
| `projects/drone_pipeline/` | Project 1 starter package: PnP perception and EKF/UKF/PF navigation for a quadrotor |
| `projects/victoria_park_slam/` | Project 2 starter package: FastSLAM on the Victoria Park dataset |
| `projects/ins_gnss/` | Project 3 starter package: ESKF INS/GNSS integration on smartphone MEMS data |
| `scripts/` | Dataset download (`fetch_data.py`) and instructor packaging (`package_datasets.py`) |
| `data/` | Downloaded datasets (not in git); see `data/README.md` |
| `tests/` | Tests for the repository scripts |
| `slides/` | Lecture slide tooling (to be replaced by Quarto reveal.js decks) |
| `admin/` | Instructor-only material: planning notes, old syllabi, templates, solutions |

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv sync
```

This creates `.venv/` with Python 3.11 or newer and installs all three project
packages in editable mode, plus the tools needed to run the book's code.

## Getting the data

```bash
uv run python scripts/fetch_data.py --list
uv run python scripts/fetch_data.py all
```

Datasets are installed under `data/<name>/`. Set `ADVNAV_DATA_DIR` to keep them
elsewhere. If a download link is not available yet, the script points to the
GitHub issue tracking it and explains how to install an archive you received
another way (`--archive`).

## Running the tests

```bash
uv run pytest
```

Tests that need a dataset are skipped, with a reason, until it is downloaded.

## Building the book

The book needs [Quarto](https://quarto.org/docs/get-started/) 1.8 or newer,
installed separately.

```bash
cd book
uv run quarto render --to html
```

The site is written to `book/output/` (not in git). Code-cell results are frozen
in `book/_freeze/` (committed), so the book builds without the datasets. After
editing a chapter's code, re-render it with the data present and commit the
updated `_freeze/` files.

## Instructor tasks

**Publishing the datasets.** Build the upload archives from the original data in
`data/`:

```bash
uv run python scripts/package_datasets.py all
```

Upload each zip from `data/_upload/` to the shared course folder on Google
Drive or OneDrive, share each *file* as "anyone with the link can view", and
paste its link into `url` in `scripts/data_sources.toml`. The archives are
built deterministically, so the recorded `sha256` and `size_bytes` stay valid
unless the data changes; if it does, copy the new values printed by the script.

**Solutions** for projects and exercises live in `admin/solutions/` and must
never be copied into the student-facing book or starter packages.

## Credits

Dataset credits are listed in `data/README.md`.
