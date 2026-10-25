# Local workflows for the course. Run `just` to list them; needs `just`, `uv`,
# and Quarto 1.8 or newer on the PATH. See README.md for the details.

# Show the available recipes.
default:
    @just --list

# Install the Python environment (creates .venv/).
setup:
    uv sync

# Run the test suite.
test:
    uv run pytest

# Download a dataset into data/<name>/ (or "all"). The script's --list shows the names.
data name="all":
    uv run python scripts/fetch_data.py {{name}}

# Render the book website (html) into book/output/.
site:
    cd book && uv run quarto render --to html

# Render the book PDF into book/output/.
pdf:
    cd book && uv run quarto render --to pdf

# Separate calls can make the html render wipe the PDF from book/output.

# Render the book website and PDF in one pass, as CI does.
book:
    cd book && uv run quarto render

# Render the lecture slide decks (html) into slides/output/.
slides-html:
    cd slides && uv run quarto render

# Needs Node.js (npx) and pdfunite (poppler-utils).

# Render the slide decks and print each to PDF, plus the combined PDF.
slides:
    scripts/build_slides.sh

# Everything CI publishes: the tests, the book, and the slides.
all: test book slides

# Preview the book with live reload while editing.
preview:
    cd book && uv run quarto preview

# Remove rendered output (book/output, slides/output). Frozen results are kept.
clean:
    rm -rf book/output slides/output
