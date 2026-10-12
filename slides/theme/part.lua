-- Colour a deck by its book part (front matter `part: N`).
--
-- Marks the page as <html data-part="N"> so theme/course.scss can pick the
-- part's colours, and gives every chapter section slide (level-1 heading) the
-- part colour as its background. The attribute is on the root element, so the
-- colours also apply when the deck is printed to PDF.

function Meta(meta)
  if meta.part then
    local part = pandoc.utils.stringify(meta.part)
    quarto.doc.include_text("in-header",
      '<script>document.documentElement.dataset.part = "' .. part .. '";</script>')
  end
end

function Header(el)
  if el.level == 1 and not el.attributes["background-color"] then
    el.attributes["background-color"] = "var(--part-accent)"
    el.classes:insert("part-section")
    return el
  end
end
