# Advanced Autonomous Navigation lecture slides

`advanced-autonomous-navigation-lecture-slides.pptx` is a lightweight, editable course deck derived from `book/`. It contains a title slide, a short usage slide, section dividers, and one prompt-oriented slide for each of the 31 teaching chapters.

The detailed book chapters remain the presenter script. Each slide has speaker notes naming its corresponding book source.

## Regenerate

The deck is generated with the bundled presentation runtime. From the repository root:

```bash
RUNTIME_NODE="/Users/jamesbrodovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
RUNTIME_NODE_MODULES="/Users/jamesbrodovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
RUNTIME_BIN_DIR="/Users/jamesbrodovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override"
ln -sfn "$RUNTIME_NODE_MODULES" slides/node_modules
"$RUNTIME_NODE" slides/generate_lecture_deck.mjs
```

Edit the `chapters` list in `generate_lecture_deck.mjs` to change slide copy or add a chapter. The PPTX remains editable in PowerPoint, Keynote, or LibreOffice.
