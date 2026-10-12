# Advanced Robot Navigation lecture slides

One [reveal.js](https://quarto.org/docs/presentations/revealjs/) deck for each
part of the book (the Technical Reference appendix has no deck). CI renders
them, prints each to PDF, and publishes both at
<https://jbrodovsky.github.io/advanced_navigation/slides/>.

| Deck | Book part |
|---|---|
| `decks/01-introduction.qmd` | Introduction |
| `decks/02-foundations.qmd` | Foundations |
| `decks/03-sensing.qmd` | Sensing and Perception (with Project 1 Parts A and B) |
| `decks/04-filtering.qmd` | Filtering and State Estimation (with Project 1 Part C and the Project 1 overview) |
| `decks/05-slam.qmd` | Simultaneous Localization and Mapping (with Project 2) |
| `decks/06-global.qmd` | Global Scale Navigation (with Project 3) |
| `decks/07-practical.qmd` | Practical Implementation Guide |
| `decks/08-advanced.qmd` | Advanced Topics |

`decks/04-filtering.qmd` is the reference deck: when in doubt, do what it does.

## Slides are written, not converted

The book is the lecture script. It explains, derives, and hedges in full
sentences. Slides are what the room looks at while the lecturer talks: the
question, the picture, the one equation, the punchline. So the decks are
authored separately from the chapters and never generated from their prose.

They do have to agree with the book. Each deck lists the chapters it covers in
its front matter:

```yaml
book-sources:
  - book/chapters/03-filtering/kf.qmd
  - book/assignments/m1-drone-pipeline-c.qmd
```

`tests/test_slides.py` checks that every teaching chapter belongs to exactly one
deck. When a chapter changes, review the deck that lists it: notation, numbers,
equations, claims, and figure code must still match. When a slide changes,
check it against the chapter.

## Deck anatomy

Front matter:

```yaml
---
title: "Filtering and State Estimation"
subtitle: "Part 4 · Advanced Robot Navigation"
part: 4                       # selects the part colour and the "Part 4" badge
footer: "Advanced Robot Navigation · Filtering and State Estimation"
book-sources: [...]
---
```

Shared settings live in `decks/_metadata.yml`; the theme is `theme/course.scss`,
and `theme/part.lua` applies the part colour.

Order of slides:

1. Title slide (automatic), then an agenda slide listing the chapters.
2. For each chapter, a level-1 heading. It becomes a full-bleed section slide in
   the part colour. Put a one-line hook and a link to the chapter on the website
   on it.
3. The chapter's content slides (level-2 headings), opening with its big
   question (`[...]{.question}`).
4. A short "in one slide" summary of the part, then a "Next: Part N+1" slide
   linking to the next deck.

Links to the book use the published site, for example
`https://jbrodovsky.github.io/advanced_navigation/chapters/03-filtering/kf.html`.
Book cross-references (`@sec-...`) and citations do not work in the decks: name
the source in words ("Kalman, 1960") and leave the bibliography to the book.

## Writing a slide

- **One idea per slide.** If the heading needs "and", it is two slides.
- **No paragraphs.** Six short lines at most. Sentence fragments are fine.
- **Show, then say.** Prefer a figure, an equation, or a table to a list.
- **Reveal in beats.** A slide that builds an argument reveals it step by step
  (recipes below). A reference slide (a table, a summary) can appear at once.
- **Equations:** one display equation per beat. Use the book's notation exactly
  ($\hat{\mathbf{x}}_{k|k-1}$, $\mathbf{P}$, $\boldsymbol{\nu}$, bold for vectors
  and matrices).
- **Code:** short excerpts (about 12 lines at most) that teach something; walk
  through them with line highlighting. Long implementations stay in the book.
  Code does not wrap inside a `.column`: keep lines short there (about 50
  characters in a half-width column).
- **Figures:** generate them with code cells (code hidden by default). Keep the
  code self-contained and fast. Figures render at 150 dpi and are scaled down to
  the width of the slide (about 1200 px) or of their column, keeping their
  aspect ratio. So the **aspect ratio sets the height**: a full-width figure
  shows about 1200 / (width / height) px tall. Leave room for the heading
  (about 110 px) and any text: about 3:1 (e.g. 12 x 4 in) with a line or two of
  text below, no taller than 2.3:1 (e.g. 11.5 x 5 in) on a slide with only a
  heading. The width in inches sets the text size: with `font.size` 13 (the
  reference deck's setup cell), a 9-12 in figure reads well; use 14-16 for
  wider figures. Prefer one row of panels to a stack. Figures that need a
  dataset are fine; their output is frozen.
- **Diagrams:** draw them with matplotlib, not mermaid (mermaid renders small,
  clipped labels on slides). For a drawing with no axes, use
  `ax = fig.add_axes([0, 0, 1, 1])` and `ax.axis("off")`; turning the axes off
  on `plt.subplots` makes the cropped figure about a quarter narrower.
  Equal-aspect drawings take their height from the data limits, not from
  `figsize`.
- **Check the fit.** Nothing may run into the footer. Render and page through
  the deck (or its PDF) before committing.
- **Speaker notes** (`::: {.notes}`): what to say that is not on the slide, the
  question to ask the room, and the book section the slide comes from.
- **Tone:** the book's light tone carries over. A joke per chapter is plenty.
- **Projects:** briefing slides only (goals, interfaces, requirements, pitfalls).
  Never solution code; nothing from `admin/`.

## Reveal recipes

Pause between beats (anything after the pause appears on the next click):

```markdown
Prediction spreads the belief.

. . .

The update concentrates it.
```

Lists that build one item at a time:

```markdown
::: {.incremental}
- First point
- Second point
:::
```

Fragments on any element, with a style:

```markdown
::: {.fragment .fade-in}
Appears.
:::

[dims once the next item appears]{.fragment .semi-fade-out}
[turns the part colour]{.fragment .highlight-accent}
[turns blue only while current]{.fragment .highlight-current-blue}
```

`fragment-index=N` sets the order explicitly when several fragments should
appear together.

A derivation that morphs from slide to slide (`auto-animate` matches elements
with the same `data-id` and animates the change):

```markdown
## Conditioning {auto-animate=true}

::: {data-id="joint"}
$$ ... joint Gaussian ... $$
:::

## Conditioning {auto-animate=true}

::: {data-id="joint"}
$$ ... joint Gaussian ... $$
:::

$$ ... the conditional ... $$
```

Walk through code one region at a time (all lines, then 2-4, then 6, then all
lines again, so the PDF, which shows the last step, is not dimmed):

````markdown
```{.python code-line-numbers="|2-4|6|1-8"}
...
```
````

Build up a figure in layers: stack several cell outputs, all the same size, and
reveal them in turn (the PDF shows the last layer). Define the plotting helper in
an earlier `#| include: false` cell. Figures are cropped to their content, so
every layer must draw the same elements: draw the not-yet-revealed ones with
`alpha=0` (or keep identical axes, labels and legend) so that the layers line up.

````markdown
::: {.r-stack}
```{python}
plot_step(0)
```
::: {.fragment}
```{python}
plot_step(1)
```
:::
:::
````

Show a code cell's output only after a click: `#| echo: true` plus
`#| output-location: fragment`.

Two columns:

```markdown
:::: {.columns}
::: {.column width="50%"}
...
:::
::: {.column width="50%"}
...
:::
::::
```

Boxes and helpers from the theme: `::: {.key}` (the one thing to remember, in
the part colour), `::: {.aside-box}` (a side remark), `[word]{.accent}` (part
colour, bold), `::: {.smaller-text}` (for a big table), `.muted`, and
`[...]{.question}` (a chapter's opening question).

Markdown does not work inside `$$ ... $$`: to reveal the terms of an equation one
at a time, show the equation and then reveal one line about each term.

`. . .` pauses only work at the top level of a slide. Inside a `.column` or any
other div, wrap the later content in `::: {.fragment}` instead. A cell with
`#| output: asis` that prints a list must print the whole
`::: {.incremental}` block itself.

decktape logs a harmless `Uncaught TypeError ... 'Config'` page error while
printing; the PDF, maths included, is fine.

Images from the book are under `images/` (a link to `book/images/`): from a deck
write `../images/name.png`. Quarto stretches a lone image to fill the rest of
its slide; when the slide also has text (which the PDF shows all at once), add
`.nostretch` and a width: `![](../images/imu.png){.nostretch width=40%}`.

Big tables: `::: {.smaller-text}` fits about six rows plus a key box. Split a
longer table across slides, ideally by a question each part answers.

## Rendering

Code cells run once, on a machine with the datasets, and their output is frozen
in `_freeze/` (committed), exactly like the book. After changing a deck's code,
render it with the data present and commit its `_freeze/` files:

```bash
uv run quarto render slides/decks/04-filtering.qmd
```

To live-edit a deck in a browser: `uv run quarto preview slides/decks/04-filtering.qmd`.

To build everything as CI does, including the PDFs (needs Node.js and
poppler-utils):

```bash
scripts/build_slides.sh
```

The PDFs come from [decktape](https://github.com/astefanutti/decktape), which
prints each slide once with all its fragments shown.

## Presenting

**F** full screen, **S** speaker view with notes and timer, **M** slide menu,
**O** overview, **B** blackboard, **C** draw on the slide, **?** all keys.
