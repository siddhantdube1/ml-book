// Single source of truth for the LLM book's chapter order, slugs, and titles.
// Mirrors docs/TRILOGY-DESIGN.md §7. `BUILT` lists the slugs that have shipped
// pages; the landing page links those and shows the rest as forthcoming.

import type { ChapterMeta } from '@trilogy/book-kit/nav'

export const chapters: ChapterMeta[] = [
  { num: 1, slug: '1-what-is-a-language-model', title: 'What is a language model?' },
  { num: 2, slug: '2-tokenization', title: 'Tokenization' },
  { num: 3, slug: '3-embeddings', title: 'Embeddings: meaning as geometry' },
  { num: 4, slug: '4-attention', title: 'Attention' },
  { num: 5, slug: '5-multi-head-self-attention', title: 'Multi-head self-attention' },
  { num: 6, slug: '6-the-transformer-block', title: 'The transformer block' },
  { num: 7, slug: '7-depth', title: 'Depth' },
  { num: 8, slug: '8-positional-information', title: 'Positional information' },
  { num: 9, slug: '9-the-forward-pass', title: 'The full forward pass' },
  { num: 10, slug: '10-pretraining', title: 'Pretraining' },
  { num: 11, slug: '11-scaling-laws', title: 'Scaling laws' },
  { num: 12, slug: '12-fine-tuning', title: 'Fine-tuning and instruction-tuning' },
  { num: 13, slug: '13-preferences-rlhf', title: 'Learning from preferences' },
  { num: 14, slug: '14-sampling', title: 'Sampling and decoding' },
  { num: 15, slug: '15-context-window', title: 'The context window' },
  { num: 16, slug: '16-prompting', title: 'Prompting and in-context learning' },
  { num: 17, slug: '17-hallucination', title: 'Why models hallucinate' },
  { num: 18, slug: '18-evaluation', title: 'Evaluation' },
  { num: 19, slug: '19-architecture-variations', title: 'Architecture variations' },
  { num: 20, slug: '20-reasoning-models', title: 'Reasoning models' },
  { num: 21, slug: '21-interpretability', title: "What we don't understand" },
]

export type Part = { name: string; blurb: string; chapters: number[] }

export const parts: Part[] = [
  {
    name: 'Part I — From text to a language model',
    blurb: 'Tokens, vectors, and the one objective that underlies everything.',
    chapters: [1, 2, 3],
  },
  {
    name: 'Part II — The Transformer',
    blurb: 'Attention, blocks, depth — the machine itself, taken apart.',
    chapters: [4, 5, 6, 7, 8, 9],
  },
  {
    name: "Part III — How they're trained",
    blurb: 'Where the capability comes from: pretraining, scale, alignment.',
    chapters: [10, 11, 12, 13],
  },
  {
    name: 'Part IV — How they behave',
    blurb: 'Sampling, context, prompting, and why models confabulate.',
    chapters: [14, 15, 16, 17, 18],
  },
  {
    name: 'Part V — The frontier',
    blurb: 'Where the architecture and our understanding are heading.',
    chapters: [19, 20, 21],
  },
]

/** Slugs of chapters with shipped pages. Grows as chapters land. */
export const BUILT = new Set<string>([
  '1-what-is-a-language-model',
  '2-tokenization',
  '3-embeddings',
  '4-attention',
  '5-multi-head-self-attention',
  '6-the-transformer-block',
  '7-depth',
  '8-positional-information',
  '9-the-forward-pass',
  '14-sampling',
])

export function chapterByNum(num: number): ChapterMeta | undefined {
  return chapters.find((c) => c.num === num)
}
