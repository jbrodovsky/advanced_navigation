-- Show each slide's speaker notes as soon as the slide opens.
--
-- reveal.js treats notes that sit inside a fragment as notes for that one
-- fragment: the speaker view shows them only once the fragment is revealed.
-- Notes written at the end of a slide after a `. . .` pause land inside the
-- last fragment, so the script would appear only after the final click.
--
-- This filter lifts every `::: {.notes}` block on a slide, at any depth, to
-- just after the slide's heading. Content before the first pause is never
-- wrapped in a fragment, so the notes become slide notes and show on arrival.
-- Authors can keep writing notes at the end of the slide.

local function is_notes(el)
  return el.t == "Div" and el.classes:includes("notes")
end

-- A slide starts at a level-1 or level-2 heading (Quarto's default slide
-- level) or at a horizontal rule.
local function starts_slide(el)
  return (el.t == "Header" and el.level <= 2) or el.t == "HorizontalRule"
end

-- Remove the notes from one block, returning the block (or nil if the block
-- was itself a notes div) and appending the notes found to `found`.
local function pull_notes(block, found)
  if is_notes(block) then
    table.insert(found, block)
    return nil
  end
  return pandoc.walk_block(block, {
    Div = function(el)
      if is_notes(el) then
        table.insert(found, el)
        return {}
      end
    end,
  })
end

function Pandoc(doc)
  local out = pandoc.List()
  local i, blocks = 1, doc.blocks
  while i <= #blocks do
    local head = blocks[i]
    out:insert(head)
    i = i + 1
    if starts_slide(head) then
      local body, notes = pandoc.List(), {}
      while i <= #blocks and not starts_slide(blocks[i]) do
        local kept = pull_notes(blocks[i], notes)
        if kept then body:insert(kept) end
        i = i + 1
      end
      out:extend(notes)
      out:extend(body)
    end
  end
  doc.blocks = out
  return doc
end
