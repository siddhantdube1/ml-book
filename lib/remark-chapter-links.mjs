// Remark plugin: turn in-prose mentions like "Chapter 11", "Chapters 18–19",
// "Chapters 9 and 11", or "Chapter-12" into links to the relevant chapter.
//
// It runs at build time over the MDX syntax tree, so no chapter source needs
// editing. Single references ("Chapter 11") become one link spanning the whole
// phrase; multi-references ("Chapters 18–19") link each number individually.
// A chapter's references to itself are left as plain text.

import fs from 'node:fs'
import path from 'node:path'

const CHAPTERS_DIR = path.join(process.cwd(), 'app', 'chapters')

function buildSlugMap() {
  const map = new Map()
  try {
    for (const entry of fs.readdirSync(CHAPTERS_DIR)) {
      const m = /^(\d+)-/.exec(entry)
      if (m) map.set(Number(m[1]), entry)
    }
  } catch {
    /* directory missing — plugin becomes a no-op */
  }
  return map
}

const SLUGS = buildSlugMap()

// "Chapter"/"Chapters", an optional separator, a number, then any run of
// connectors (dash, "and", "to", "through", "or", comma, ampersand) + numbers.
const REF =
  /\bChapters?\b[\s-]*\d+(?:\s*(?:[–—-]|to|through|and|or|,|&)\s*\d+)*/g
const NUM = /\d+/g

function linkNode(slug, text) {
  return {
    type: 'link',
    url: `/chapters/${slug}`,
    children: [{ type: 'text', value: text }],
  }
}

function linkifyReference(text, current) {
  const numbers = text.match(NUM)
  if (!numbers) return [{ type: 'text', value: text }]

  // Single reference: link the whole "Chapter N" phrase as one unit.
  if (numbers.length === 1) {
    const n = Number(numbers[0])
    const slug = SLUGS.get(n)
    if (!slug || n === current) return [{ type: 'text', value: text }]
    return [linkNode(slug, text)]
  }

  // Multiple references: keep words and connectors as text, link each number.
  const out = []
  let last = 0
  let m
  NUM.lastIndex = 0
  while ((m = NUM.exec(text)) !== null) {
    if (m.index > last) out.push({ type: 'text', value: text.slice(last, m.index) })
    const n = Number(m[0])
    const slug = SLUGS.get(n)
    if (slug && n !== current) out.push(linkNode(slug, m[0]))
    else out.push({ type: 'text', value: m[0] })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ type: 'text', value: text.slice(last) })
  return out
}

function transformText(value, current) {
  REF.lastIndex = 0
  if (!REF.test(value)) return null
  REF.lastIndex = 0

  const out = []
  let last = 0
  let m
  while ((m = REF.exec(value)) !== null) {
    if (m.index > last) out.push({ type: 'text', value: value.slice(last, m.index) })
    out.push(...linkifyReference(m[0], current))
    last = m.index + m[0].length
  }
  if (last < value.length) out.push({ type: 'text', value: value.slice(last) })
  return out
}

function walk(node, current) {
  // Don't rewrite inside existing links or code.
  if (node.type === 'link' || node.type === 'code' || node.type === 'inlineCode') {
    return
  }
  if (!Array.isArray(node.children)) return

  for (let i = 0; i < node.children.length; i++) {
    const child = node.children[i]
    if (child.type === 'text') {
      const replaced = transformText(child.value, current)
      if (replaced) {
        node.children.splice(i, 1, ...replaced)
        i += replaced.length - 1
      }
    } else {
      walk(child, current)
    }
  }
}

export default function remarkChapterLinks() {
  return (tree, file) => {
    if (SLUGS.size === 0) return
    const filePath = file?.path || file?.history?.[0] || ''
    const cm = /chapters[\\/](\d+)-/.exec(filePath)
    const current = cm ? Number(cm[1]) : null
    walk(tree, current)
  }
}
