# Advanced Robot Navigation lecture slides

`advanced-autonomous-navigation-lecture-slides.pptx` is a lightweight, editable course deck derived from `book/`. It contains a title slide, a short usage slide, section dividers, and one prompt-oriented slide for each of the 30 teaching chapters.

The detailed book chapters remain the presenter script. Each slide has speaker notes naming its corresponding book source.

## Regenerate

The deck is generated with Node.js and the `@oai/artifact-tool` presentation
library, which `generate_lecture_deck.mjs` imports. With that library available
in `slides/node_modules/`, run from the repository root:

```bash
node slides/generate_lecture_deck.mjs
```

Edit the `chapters` list in `generate_lecture_deck.mjs` to change slide copy or add a chapter. The PPTX remains editable in PowerPoint, Keynote, or LibreOffice.
