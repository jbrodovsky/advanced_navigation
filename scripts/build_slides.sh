#!/usr/bin/env bash
# Render the lecture slides and print each deck to PDF.
#
# Usage (from anywhere): scripts/build_slides.sh
#
# Code-cell output comes from slides/_freeze/, so this works without the
# datasets, as in CI. Needs Quarto, Node.js (for npx decktape, which downloads
# its own headless Chrome on first use), and pdfunite (poppler-utils).
#
# Chrome runs without its sandbox because GitHub's Ubuntu runners do not allow
# it; it only ever opens the decks rendered here.
#
# Output in slides/output/: decks/<deck>.html to present, decks/<deck>.pdf with
# every slide fully revealed, and advanced-robot-navigation-slides.pdf with all
# the decks in part order.
set -euo pipefail

DECKTAPE_VERSION=3.16.1

cd "$(dirname "$0")/../slides"
uv run quarto render

pdfs=()
for html in output/decks/*.html; do
  pdf="${html%.html}.pdf"
  echo "Printing ${html} -> ${pdf}"
  npx --yes "decktape@${DECKTAPE_VERSION}" reveal \
    --size 1280x720 --load-pause 1000 --pause 300 \
    --chrome-arg=--no-sandbox \
    "${html}" "${pdf}"
  pdfs+=("${pdf}")
done

pdfunite "${pdfs[@]}" output/advanced-robot-navigation-slides.pdf
echo "Wrote ${#pdfs[@]} deck PDFs and output/advanced-robot-navigation-slides.pdf"
