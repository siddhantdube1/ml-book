# Interactive ML Textbook — Handoff Document

> **For Claude Code:** read this file in full before making changes. It is the single source of truth for the project's structure, conventions, and the voice the book is written in. Companion files in `docs/` give focused recipes for the most common tasks.

---

## Quick reference

| Thing | Value |
|---|---|
| **Live URL** | https://ml-book-seven.vercel.app |
| **Repo** | The directory you are reading (Vercel auto-deploys on `git push`) |
| **Tech stack** | Next.js 15 App Router · React 18 · TypeScript · MDX · Tailwind 3 · KaTeX · Pyodide 0.26.4 |
| **Shipped chapters** | 6 (Gradient descent) · 7 (Logistic regression) · 18 (k-means clustering) |
| **Total planned** | 22 chapters across 7 parts |
| **Pyodide** | numpy preloaded automatically; matplotlib/scipy/sklearn available on demand |

**Deployment workflow:**
```bash
git add -A
git commit -m "Chapter N: <title>"
git push
# Vercel deploys in ~2 minutes
```

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Tech stack & file layout](#2-tech-stack--file-layout)
3. [Design system (CSS variables, fonts, colors)](#3-design-system)
4. [Writing voice & style](#4-writing-voice--style)
5. [Chapter MDX template (the recipe)](#5-chapter-mdx-template)
6. [Visualization component patterns](#6-visualization-component-patterns)
7. [Established library modules](#7-established-library-modules)
8. [Pyodide & editable code](#8-pyodide--editable-code)
9. [Problems section pattern](#9-problems-section-pattern)
10. [Math notation conventions](#10-math-notation-conventions)
11. [Build / dev / deploy workflow](#11-build--dev--deploy-workflow)
12. [Smoke test checklist for new chapters](#12-smoke-test-checklist)
13. [Pitfalls already encountered (and fixed)](#13-pitfalls)
14. [Roadmap — next chapters](#14-roadmap)
15. [Tips for productive Claude Code sessions](#15-tips-for-claude-code)

---

## 1. Project overview

This is an interactive machine-learning textbook in the spirit of cartesian.app — a web app where every chapter combines literate, opinionated prose with interactive visualizations and runnable Python (via Pyodide in the browser, no backend). The reader can drag handles, scrub through playback timelines, change parameters, and execute code, all inside the same continuous reading experience.

The book covers **classical ML, end-to-end**, in 22 chapters organized into 7 parts:

```
Part I    — Foundations            (Ch 1-4)
Part II   — Linear models          (Ch 5-9)    ← Ch 6, 7 shipped
Part III  — Evaluating models      (Ch 10-11)
Part IV   — Tree-based models      (Ch 12-14)
Part V    — Other classical models (Ch 15-17)
Part VI   — Unsupervised learning  (Ch 18-20)  ← Ch 18 shipped
Part VII  — Neural networks        (Ch 21-22)
```

Each chapter is roughly 3,500 words of prose, 3 interactive visualizations, runnable Python, and 5 problems with collapsible solutions. The shipped chapters (18, 6, 7) establish the format and have proven it works across clustering, optimization, and classification — three distinct ML domains. Remaining chapters follow the same recipe.

The author (Siddhant) is an AI researcher who is writing this both as a pedagogical project and a portfolio piece. The voice is opinionated, technical, beginner-friendly without being condescending. **Match the voice exactly** — see [section 4](#4-writing-voice--style) for verbatim examples.

---

## 2. Tech stack & file layout

### Stack

- **Next.js 15 App Router** with static rendering for all chapter pages
- **React 18** + **TypeScript** (strict mode, no `any`)
- **MDX** (`@next/mdx`) with `remark-math` and `rehype-katex` for inline LaTeX
- **Tailwind 3** with custom CSS variables (no `@apply` overuse, no design tokens in arbitrary brackets — see [section 3](#3-design-system))
- **KaTeX** for math rendering (the CSS is imported in `app/layout.tsx`)
- **Pyodide v0.26.4** loaded from `cdn.jsdelivr.net`, with **numpy preloaded** on first init
- **CodeMirror 6** for code editor UI inside `PyodideEditor`
- **Vercel** for deployment (zero-config)

### Directory tree (key files only)

```
ml-book/
├── HANDOFF.md                         ← you are here
├── docs/
│   ├── CHAPTER-RECIPE.md              ← step-by-step recipe for a new chapter
│   └── VOICE-AND-STYLE.md             ← verbatim voice examples
├── app/
│   ├── layout.tsx                     ← <html>, fonts, KaTeX CSS, base metadata
│   ├── page.tsx                       ← landing page (TOC of all 22 chapters)
│   ├── globals.css                    ← design tokens, base typography
│   ├── icon.svg                       ← three-cluster favicon
│   ├── robots.ts                      ← generates robots.txt
│   ├── sitemap.ts                     ← generates sitemap.xml; add new chapters here
│   └── chapters/
│       ├── layout.tsx                 ← shared chapter shell with "Table of contents" back-link
│       ├── 6-gradient-descent/
│       │   └── page.mdx
│       ├── 7-logistic-regression/
│       │   └── page.mdx
│       └── 18-k-means/
│           └── page.mdx
├── components/
│   ├── PyodideEditor.tsx              ← in-browser Python editor + Run button
│   ├── Tex.tsx                        ← KaTeX renderer for use inside .tsx (NOT in MDX prose)
│   │
│   │  -- Chapter 18 widgets --
│   ├── KMeansPlayback.tsx
│   ├── UpdateStepDemo.tsx
│   ├── ConvergencePlot.tsx
│   ├── KSweepWidget.tsx
│   ├── InitComparisonDemo.tsx
│   │
│   │  -- Chapter 6 widgets --
│   ├── GradientDescent1D.tsx
│   ├── GradientDescent2D.tsx
│   ├── LRComparison.tsx
│   │
│   │  -- Chapter 7 widgets --
│   ├── SigmoidExplorer.tsx
│   ├── DecisionBoundary.tsx
│   └── TrainingDynamics.tsx
├── lib/
│   ├── rng.ts                         ← createRng(seed), gauss(rng, mu, sigma)
│   ├── datasets.ts                    ← unlabeled clustering datasets (blobs/aniso/moons)
│   ├── kmeans.ts                      ← k-means algorithm with frame history
│   ├── silhouette.ts                  ← clustering evaluation metrics
│   ├── gradient.ts                    ← loss functions + runOptimizer (GD/momentum/SGD)
│   ├── contours.ts                    ← marching squares contour generation
│   ├── logistic.ts                    ← sigmoid, model, cross-entropy, trainLogistic, labeled data
│   └── pyodide.ts                     ← singleton Pyodide loader (numpy preloaded)
├── next.config.mjs                    ← MDX + remark-math + rehype-katex setup
├── tailwind.config.ts                 ← Tailwind theme; references CSS variables
├── tsconfig.json
└── package.json
```

### NPM scripts

```bash
npm run dev       # next dev — for local development (http://localhost:3000)
npx next build    # production build; use for verification
npx next start    # serve the production build locally (for smoke tests)
```

---

## 3. Design system

The design system lives in `app/globals.css`. **Do not introduce new colors or fonts in component code** — always reference the CSS variables below.

### Fonts

| Variable | Family | Use case |
|---|---|---|
| `--font-serif` | **Source Serif 4** | All body prose (default `<body>` font) |
| `--font-sans` | **Geist** | UI elements: buttons, labels, captions, axis text, controls |
| `--font-mono` | **JetBrains Mono** | Code blocks, numeric readouts in widgets |

Use the Tailwind classes: `font-sans`, `font-mono` for sans / mono. Serif is the default and shouldn't need to be set explicitly.

### Colors (light mode; dark mode auto-flips via `prefers-color-scheme`)

| Variable | Light hex | Use case |
|---|---|---|
| `--paper` | `#fbf8f1` (warm off-white) | Background |
| `--paper-soft` | `#f4eee2` | Widget container backgrounds |
| `--ink` | `#1a1a1a` (near-black) | Main text, graph lines |
| `--ink-muted` | `#666` | Captions, axis labels, secondary text |
| `--rule` | `#d9d2c0` | Borders, separators, axes |
| `--accent` | `#1d6d5e` (teal) | Highlight color: trajectory paths, current-state dots, key annotations |

### Class colors (for binary classification visuals)

These are hard-coded in component code — they aren't CSS variables yet but should remain consistent across the book:

| Class | Color | Use |
|---|---|---|
| **Class 0** | `#3c5a8c` (blue) | "Negative" class points, leftward heatmap shading |
| **Class 1** | `#c7522a` (orange) | "Positive" class points, rightward heatmap shading |

### Spacing & layout conventions

- Widget container: `border border-rule rounded-lg overflow-hidden bg-paper-soft`
- SVG inside container: `viewBox` set, `className="w-full h-auto block"`
- Standard SVG dimensions for 1-panel widgets: ~640×320 or 720×420 (viewBox)
- Two-panel widgets: split via flexbox — see `TrainingDynamics.tsx`
- Padding inside SVG: PAD = 24–40 depending on widget
- Border-radius: `rounded-lg` (Tailwind) for containers; `rounded` for buttons
- Buttons: `px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors`
- Sliders: `w-32` to `w-48` (varies by what makes sense), with label format:
  ```tsx
  <label className="flex items-center gap-2">
    <span className="text-ink-muted">
      Learning rate = <span className="font-mono">{lr.toFixed(3)}</span>
    </span>
    <input type="range" ... />
  </label>
  ```

### Figure captions

Every widget ends with:
```tsx
<figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
  Figure N.M — Description that says what the reader is looking at and what they should notice.
</figcaption>
```

Number figures `<Chapter>.<Index>`: Chapter 6's first figure is 6.1, second is 6.2, etc.

---

## 4. Writing voice & style

**Read `docs/VOICE-AND-STYLE.md` for full verbatim examples.** Summary:

The voice is **confident, direct, intuition-first, never condescending**. Some specific habits:

- **Open with a sensory or physical metaphor** before going formal. ("Picture being dropped onto a hilly landscape, blindfolded." "Imagine sorting laundry without being told what the categories are.")
- **State the thing, then justify it.** Not: "We could perhaps consider that maybe..."
- **British spelling** (visualise, optimise, modelling, behaviour).
- **Em-dashes** are common but not abused.
- **Conversational asides** are fine and welcome ("That's it. That's the entire training algorithm.").
- **No bullet point spam.** Use prose for ideas; lists only when the items are genuinely parallel.
- **Math is grounded in geometric intuition** before formalism.
- **Cross-reference other chapters** by section: "We'll see this fixed in §6 with momentum" or "preview of Chapter 17" or "this is exactly what we built in Chapter 6."
- **Foreshadow the rest of the book.** Each chapter ends with "*Next: Chapter N+1 — Title.*" plus a one-sentence hook.
- **Use the second person ("you")** when guiding through interaction or thought experiment, third person when stating facts.
- **End sections with a one-line clinch** — not a summary, a sharp closing thought.

Anti-patterns to avoid:

- ❌ Tutorial-speak: "In this section we will learn..."
- ❌ Hedging: "It might be the case that..."
- ❌ Over-formality: "We shall now proceed to derive..."
- ❌ Lists of bullets where prose would carry the same information with style
- ❌ Apologetic clarification: "(Don't worry if this doesn't make sense yet...)"
- ❌ Marketing voice: "powerful," "elegant," "groundbreaking"

---

## 5. Chapter MDX template

Every chapter file is at `app/chapters/<num>-<slug>/page.mdx`. The structure is rigid — match it exactly for new chapters. Here's the skeleton with annotations:

```mdx
import GradientDescent1D from '@/components/GradientDescent1D'    // ← import each widget
import PyodideEditor from '@/components/PyodideEditor'             // ← always import this for §10 + §11

// EXPORT METADATA — used by Next.js for <title> and OG tags.
// title becomes "Logistic regression · The Interactive Handbook on Machine Learning"
// description is what appears in Slack/Twitter/LinkedIn previews
export const metadata = {
  title: 'Logistic regression',
  description:
    'From predicting numbers to predicting probabilities. The sigmoid, cross-entropy loss, and the moment the gradient descent we just built starts training real classifiers — with three interactive visualisations, runnable Python, and five problems.',
  openGraph: {
    title: 'Logistic regression',
    description:
      'An interactive walk-through of logistic regression — sigmoid, cross-entropy, and gradient descent on a real binary classifier.',
    type: 'article',
  },
}

// CHAPTER LABEL — small uppercase tracking line above the title
<p className="font-sans text-xs uppercase tracking-[0.18em] text-ink-muted mb-3">
  Chapter 7 · Part II — Linear models
</p>

# Logistic regression

*Predicting probabilities instead of numbers. The moment the gradient
descent we just built starts training real classifiers.*

[Opening paragraph — sensory hook or motivating example.
2–4 sentences. Set the stage.]

[Second paragraph — what this chapter teaches and why.]

[Third paragraph — the promise: "By the end of this chapter..."]

## 1. <Section title — sentence case>

[Section prose. ~300–400 words. End with a transition into §2.]

## 2. <Section title>

[Embed widget if applicable]

<SigmoidExplorer />

[Optional follow-up prose after a widget — usually a bulleted list of things
to play with, in markdown.]

## 3. <Section>
...

## 10. Implementing it yourself

[Brief lead-in.]

<PyodideEditor
  initialCode={`import numpy as np
# 20-30 lines of from-scratch implementation
`}
/>

## 11. Problems

### Problem 1 — <Title>

[Problem statement — visible.]

<PyodideEditor initialCode={`# starter code with placeholders`} />

<details>
  <summary>Show solution</summary>

<PyodideEditor initialCode={`# full solution code`} />

[Optional explanation paragraph after the solution.]

</details>

[Repeat for Problems 2–5.]

---

*Next: Chapter N+1 — <title>.* One-sentence hook that previews the next chapter's idea.
```

### Required 11-section structure

This structure is established across all three shipped chapters. Stick to it:

1. **Conceptual setup** — what is this problem, what's the goal
2. **The naive approach and why it fails** (or: the central tension)
3. **The key idea** — the new concept that makes this chapter's algorithm work
4. **The model / algorithm definition** (often with widget)
5. **Loss / criterion**
6. **Training / optimization** (often with main widget showing dynamics)
7. **Geometric / structural consequence** of the algorithm
8. **Practical concerns** — confidence, calibration, edge cases
9. **Complexity** — Big-O cost per step, convergence
10. **Implementation from scratch** in NumPy with PyodideEditor
11. **Problems** — 5 problems with collapsible solutions

Numbers and exact titles can vary, but the *shape* of the arc — concept → motivation → definition → loss → training → consequence → practical → complexity → code → problems — should be preserved. The reader who reads two chapters in a row should feel the structure is familiar.

### Section length guidance

- Total prose: ~3,500 words
- §1: ~250 words (the hook + setup)
- §2-§8: ~300-500 words each (the meat)
- §9 (complexity): ~200 words (terse)
- §10 (implementation): ~150 words intro + ~25 lines code
- §11 (problems): five problems, each ~50 words of prompt + starter code + solution

---

## 6. Visualization component patterns

### The frame-history playback pattern

This is the central pattern. **Every widget that shows an iterative algorithm follows it.** It works because most ML algorithms are iterative — they take steps. We precompute all the steps as a `Frame[]` array, then render the frame at the current index, with playback controls to step through.

```tsx
'use client'

import { useEffect, useMemo, useState } from 'react'
import { runMyAlgorithm, type Frame } from '@/lib/my-algorithm'

const W = 640    // SVG viewBox width
const H = 360    // SVG viewBox height
const PAD = 30   // padding inside viewBox

export default function MyWidget() {
  // === State ===
  const [param, setParam] = useState(defaultValue)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  // === Compute all frames upfront ===
  const frames: Frame[] = useMemo(
    () => runMyAlgorithm(input, options),
    [input, options],  // recompute when inputs change
  )

  // === Reset frame & playback when frames change ===
  useEffect(() => {
    setFrameIdx(0)
    setPlaying(false)
  }, [frames])

  // === Playback animation ===
  useEffect(() => {
    if (!playing) return
    if (frameIdx >= frames.length - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrameIdx((i) => i + 1), 250)  // ms per frame
    return () => clearTimeout(t)
  }, [playing, frameIdx, frames.length])

  // === Coordinate transforms (domain → screen) ===
  const sx = (x: number) => PAD + ((x - xMin) / (xMax - xMin)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y - yMin) / (yMax - yMin)) * (H - 2 * PAD)

  const cur = frames[Math.min(frameIdx, frames.length - 1)]

  return (
    <figure className="my-10">
      {/* Controls row */}
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        {/* ... labels with sliders ... */}
      </div>

      {/* SVG canvas */}
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* ... background, data, trajectory, current state ... */}
        </svg>
      </div>

      {/* Playback row */}
      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => {
          if (frameIdx >= frames.length - 1) setFrameIdx(0)
          setPlaying((p) => !p)
        }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[60px]">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={() => { setFrameIdx(0); setPlaying(false) }} className="...">Reset</button>
        <button onClick={() => setFrameIdx(i => Math.min(i + 1, frames.length - 1))} className="...">Step</button>
        <input
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 1)}
          value={frameIdx}
          onChange={(e) => { setPlaying(false); setFrameIdx(parseInt(e.target.value)) }}
          className="flex-1"
        />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">
          step {frameIdx} / {frames.length - 1}
        </span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>...</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure N.M — ...
      </figcaption>
    </figure>
  )
}
```

### Coordinate transforms

Always work in **math/data coordinates** in computation and transform to screen at render time. Math coords have y going **up**; SVG has y going **down** — the transform handles the flip.

```ts
// math/data → screen
const sx = (x: number) => PAD + ((x - xMin) / (xMax - xMin)) * (W - 2 * PAD)
const sy = (y: number) => H - PAD - ((y - yMin) / (yMax - yMin)) * (H - 2 * PAD)

// screen → math/data (for click-to-place, drag handlers)
const invX = (px: number) => xMin + ((px - PAD) / (W - 2 * PAD)) * (xMax - xMin)
const invY = (py: number) => yMin + ((H - PAD - py) / (H - 2 * PAD)) * (yMax - yMin)
```

### Click-to-place pattern

```tsx
const svgRef = useRef<SVGSVGElement>(null)

function handleClick(e: React.MouseEvent<SVGSVGElement>) {
  const svg = svgRef.current
  if (!svg) return
  const pt = svg.createSVGPoint()
  pt.x = e.clientX
  pt.y = e.clientY
  const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
  // Now (x, y) are in SVG viewBox coords
  if (x < PAD || x > W - PAD || y < PAD || y > H - PAD) return
  // Convert to domain coords
  setSomething([invX(x), invY(y)])
}

return <svg ref={svgRef} onClick={handleClick} className="cursor-crosshair" ...>
```

### Draggable handle pattern (with pointer capture)

```tsx
const [dragging, setDragging] = useState(false)

function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
  if (!dragging) return
  // ... compute new position, setState
}

<svg
  ref={svgRef}
  onPointerMove={onPointerMove}
  onPointerUp={(e) => {
    setDragging(false)
    e.currentTarget.releasePointerCapture(e.pointerId)
  }}
>
  <circle
    cx={sx(handlePos[0])} cy={sy(handlePos[1])} r={8}
    fill="var(--paper)" stroke="var(--ink)" strokeWidth={2}
    className="cursor-grab"
    onPointerDown={(e) => {
      setDragging(true)
      e.currentTarget.setPointerCapture(e.pointerId)
      e.stopPropagation()
    }}
  />
</svg>
```

### Heatmap rendering (for probability/density fields)

Sample on a coarse grid, render as filled `<rect>` elements. 40×30 = 1200 rectangles renders smoothly. See `DecisionBoundary.tsx` for the canonical example.

### Contour plot rendering

Use `computeContours()` from `lib/contours.ts` — marching squares. See `GradientDescent2D.tsx` for the canonical example.

### When in doubt, copy the most similar existing widget

The three Chapter 7 widgets cover the three main shapes:
- **`SigmoidExplorer.tsx`** — single static curve + draggable line, no playback
- **`DecisionBoundary.tsx`** — interactive scene with draggable handles + heatmap, no playback
- **`TrainingDynamics.tsx`** — two-panel playback (data space + loss curve)

The three Chapter 6 widgets cover the optimization-trajectory shapes:
- **`GradientDescent1D.tsx`** — function curve + descending dot + playback
- **`GradientDescent2D.tsx`** — contour plot + trajectory + click-to-place + playback
- **`LRComparison.tsx`** — multiple synchronized trajectories overlay

The five Chapter 18 widgets cover the discrete-state algorithm shapes (k-means iterations).

---

## 7. Established library modules

### `lib/rng.ts`

Seedable PRNG. Use this any time you need randomness — never `Math.random()` directly.

```ts
import { createRng, gauss } from '@/lib/rng'

const rng = createRng(42)
const u = rng()              // uniform [0, 1)
const g = gauss(rng, 0, 1)   // standard normal
```

### `lib/datasets.ts`

Unlabeled clustering datasets used by Chapter 18. **Returns points in screen coordinates (0–700ish, 0–300ish).** Use only for clustering widgets that share Chapter 18's coordinate system.

### `lib/kmeans.ts`

k-means algorithm with full frame history. Returns `Frame[]` where each frame contains assignments and centroid positions per step.

### `lib/silhouette.ts`

Silhouette score for k-means quality. Used by `KSweepWidget.tsx`.

### `lib/gradient.ts`

Gradient descent library used by Chapter 6.

```ts
import { runOptimizer, LOSSES_2D, type LossFn, type GDFrame } from '@/lib/gradient'

// Unified optimizer: handles vanilla GD, momentum, SGD, or any combination
const frames: GDFrame[] = runOptimizer(BOWL_2D, [2.5, 2.2], {
  lr: 0.1,
  momentum: 0,        // > 0 for momentum
  noiseScale: 0,      // > 0 for SGD
  maxSteps: 80,
  tol: 1e-4,
})
```

Loss-function registry exports: `BOWL_2D`, `VALLEY_2D`, `SADDLE_2D`, `HIMMELBLAU`, `PARABOLA_1D`, `QUARTIC_1D`, `ABS_1D`. All have a `.domain`, `.eval`, `.grad`, optional `.globalMin`, `.localMinima`, `.contourLevels`.

### `lib/contours.ts`

Marching squares for 2D contour generation.

```ts
import { computeContours } from '@/lib/contours'

const contours = computeContours(BOWL_2D, [0.25, 1, 2.25, 4, 6.25, 9], 80)
// Returns: { level: number, segments: [x1, y1, x2, y2][] }[]
```

### `lib/logistic.ts`

Logistic regression library used by Chapter 7.

```ts
import {
  sigmoid, predictProb, predict, crossEntropy, accuracy,
  trainLogistic, makeBlobs, makeMoons, makeOverlap,
  type LabeledPoint, type LRModel, type LRFrame,
} from '@/lib/logistic'

// Numerically stable sigmoid
const p = sigmoid(z)

// Train with frame history for playback
const data: LabeledPoint[] = makeBlobs(100, /* seed */ 42)
const frames: LRFrame[] = trainLogistic(data, {
  lr: 0.5,
  maxSteps: 80,
  w0: [0.1, -0.2],   // initial weights
  b0: 0.3,
})
```

Labeled-data generators: `makeBlobs(n, seed, separation=2.5, std=0.7)`, `makeOverlap(n, seed)`, `makeMoons(n, seed, noise=0.15)`. All return `LabeledPoint[]` in **math coordinates centered near origin** (range roughly [-3, 3] × [-2, 2]).

### `lib/pyodide.ts`

Singleton Pyodide loader. Calls `loadPyodideOnce()` once per session, preloads numpy automatically. Don't import this directly in chapter code — use `<PyodideEditor>` instead.

---

## 8. Pyodide & editable code

### The `<PyodideEditor>` component

Embedded Python editor with a Run button. Code executes in the browser via Pyodide. Use this for the §10 implementation block in each chapter and inside `<details>` solutions in §11 problems.

```mdx
<PyodideEditor
  initialCode={`import numpy as np

# Your code here. Multi-line strings work; escape backticks if needed
# but you usually won't have any.

print("hello")
`}
/>
```

**Key points:**

- The first PyodideEditor run on a page takes ~5 seconds (downloads Pyodide + numpy from CDN).
- All subsequent runs across all editors in the session are instant.
- **numpy is preloaded** — `import numpy as np` always works without extra config.
- For other packages (matplotlib, scipy, sklearn), pass them as a prop: `<PyodideEditor packages={['scipy']} initialCode={...} />` — adds ~3s the first time that package is loaded.
- Inside the `initialCode` template literal:
  - Use `\\n` to write a literal `\n` in the displayed code (or just press enter — multiline strings preserve newlines).
  - Escape backticks if any appear in the code (rare).
  - Tabs/spaces in indentation are preserved; use **4 spaces** (Python convention).

### Common idioms in the runnable code

Match what's already in Chapter 6 / 7 §10:

```python
import numpy as np

# Reproducible randomness
rng = np.random.default_rng(42)

# Numerically stable sigmoid (paste this any time you need it)
def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

# Progress prints every N steps
for step in range(100):
    # ...
    if step % 10 == 0:
        print(f"step {step:3d}: loss = {loss:.4f}")
```

---

## 9. Problems section pattern

Five problems per chapter. **Problem statements and starter code are always visible. Only the solution is behind a `<details>` toggle.** This was a bug I had to fix — see [section 13](#13-pitfalls).

### Problem template

```mdx
### Problem N — <Title>

[Problem statement — visible. 1–3 sentences. Concrete and specific:
"Train for 200 steps with learning rate 0.5 and report final accuracy"
not "explore how gradient descent behaves on this dataset".]

<PyodideEditor
  initialCode={`# Starter code with placeholders for the student
# Always provides the data setup and imports
# Has "Your code:" comments where the student fills in

import numpy as np

# ... setup ...

# Your code:
#   1. Step 1 description
#   2. Step 2 description
#   3. Step 3 description
`}
/>

<details>
  <summary>Show solution</summary>

<PyodideEditor
  initialCode={`# Full solution code that the student can run as-is
# Same data setup as above, plus the filled-in solution

import numpy as np

# ... full solution ...
`}
/>

[Brief explanation paragraph that explains *why* the solution works,
not just *what* it does. Often contains a fact worth knowing.]

</details>
```

### Problem ordering (heuristic)

1. **Conceptual question** — no code, tests understanding of a key idea from the chapter
2. **Implement a building block** — sigmoid, distance function, gradient computation
3. **Apply the building block** — use what you implemented in #2 inside a small loop
4. **Full algorithm** — combine everything into a working training loop
5. **Putting it together / breaking it** — apply the algorithm to a tricky dataset, or extend it. Often previews the next chapter or motivates Chapter 17 (feature engineering).

---

## 10. Math notation conventions

### Inline math

Use single-dollar delimiters in MDX prose:

```mdx
The gradient $\nabla L(\theta)$ points in the direction of steepest ascent.
```

### Display math

Use double-dollar on its own line:

```mdx
$$\theta \leftarrow \theta - \alpha \nabla L(\theta)$$
```

### Inside `.tsx` components (not MDX)

Use the `<Tex>` component, **not** raw `$...$` (KaTeX isn't auto-processed in JSX):

```tsx
import { Tex } from './Tex'

<p>The loss is <Tex>L(\theta) = \frac{1}{n} \sum_i (\hat{y}_i - y_i)^2</Tex>.</p>
```

### LaTeX style guide

- **Vectors and matrices**: lowercase bold or unadorned. `w`, `x`, `\beta`, not `\vec{w}` unless ambiguity demands it.
- **Transpose**: `w^\top` not `w^T`.
- **Estimated quantities**: `\hat{y}`, `\hat{p}`.
- **Expectations**: `\mathbb{E}[\cdot]`.
- **Real numbers**: `\mathbb{R}`.
- **Greek for parameters**: `\theta` (general), `\beta` (regression), `\alpha` (learning rate), `\beta` (momentum coefficient — context disambiguates).

---

## 11. Build / dev / deploy workflow

### Local development

```bash
cd ml-book
npm install              # first time only
npm run dev              # http://localhost:3000
```

### Production build (always run before pushing)

```bash
npx next build
```

This will fail loudly if you have:
- TypeScript errors
- Broken imports
- MDX syntax errors
- Missing dependencies

The output shows a per-route bundle size table. For comparison:
- Chapter 18 (5 widgets): ~88 kB / 346 kB First Load JS
- Chapter 6 (3 widgets): ~7 kB / 265 kB
- Chapter 7 (3 widgets): ~6.5 kB / 264 kB

If a new chapter's First Load JS suddenly jumps above 400 kB, something is being bundled wrong (likely a missing `'use client'` or an accidental import of a giant dependency).

### Smoke test against the production build

```bash
npx next start -p 3000
# Then in another shell:
curl -s http://localhost:3000/chapters/<num>-<slug> | grep -E "<title>|katex-html" | head
```

### Deploy

```bash
git add -A
git commit -m "Chapter N: <title>"
git push
```

Vercel auto-deploys on push to main. URL becomes live in ~2 minutes.

---

## 12. Smoke test checklist

After building a new chapter, verify:

```bash
# 1. The route exists and returns 200
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000/chapters/<N>-<slug>

# 2. KaTeX rendered (count katex-html elements; should be > 20 for any chapter with math)
curl -s http://localhost:3000/chapters/<N>-<slug> | grep -c "katex-html"

# 3. Problem structure: 5 h3 headings, 5 details, 5 "Show solution" summaries
curl -s http://localhost:3000/chapters/<N>-<slug> | python3 -c "
import sys, re
html = sys.stdin.read()
print('Problem headings:', len(re.findall(r'<h3[^>]*>Problem \d', html)))
print('details blocks:  ', html.count('<details'))
print('Show solution:   ', len(re.findall(r'<summary[^>]*>Show solution</summary>', html)))
"

# 4. Sitemap includes the new chapter
curl -s http://localhost:3000/sitemap.xml | grep <slug>

# 5. OG meta tags set (for sharing)
curl -s http://localhost:3000/chapters/<N>-<slug> | grep -E "og:title|og:description"
```

Expected: HTTP 200, KaTeX count > 50, 5 problem headings, 5 details, 5 Show solution summaries, sitemap match, OG tags present.

Also manually verify in browser:
- [ ] All widgets render (no React errors in console)
- [ ] Sliders/buttons all work
- [ ] PyodideEditor runs without numpy errors
- [ ] Solutions hidden until "Show solution" clicked
- [ ] Math renders correctly (no raw `\frac` showing through)
- [ ] Landing page link to new chapter works
- [ ] "Table of contents" back-link returns to landing
- [ ] Dark mode looks reasonable (toggle browser preference)

---

## 13. Pitfalls already encountered (and fixed)

These bugs cost real iteration time. Learn from them.

### 13.1 Pyodide doesn't include numpy by default

**Symptom:** `ModuleNotFoundError: No module named 'numpy'`

**Cause:** Pyodide ships numpy as an available package but doesn't install it until something calls `loadPackage('numpy')`.

**Fix:** Already applied in `lib/pyodide.ts` — `loadPyodideOnce()` preloads numpy after init. The 3-second cost is paid once per session.

**If you need another package** (matplotlib, scipy, sklearn): pass it to the editor:
```mdx
<PyodideEditor packages={['scipy']} initialCode={...} />
```
Or add to the global preload in `lib/pyodide.ts` if every chapter uses it (currently only numpy is global).

### 13.2 Problem solutions visible by default

**Symptom:** Clicking the problem title reveals both the problem and the solution at once.

**Cause:** Wrapped the entire problem inside a single `<details>` with summary = problem title. Wrong.

**Fix:** Problem title is a visible `<h3>`. Problem statement and starter code are normal markdown content (visible). Only the solution sits inside `<details><summary>Show solution</summary>`.

See `app/chapters/7-logistic-regression/page.mdx` §11 for the correct pattern. Match it exactly.

### 13.3 KaTeX not rendering inside `.tsx` components

**Symptom:** Raw `\frac{1}{n}` shows up instead of rendered math.

**Cause:** The MDX `$...$` syntax only works in `.mdx` files. Inside `.tsx`, you need the `<Tex>` component.

**Fix:** Use `<Tex>` in components, `$...$` in MDX. Don't mix.

### 13.4 SVG y-axis confusion

**Symptom:** Math plots upside down.

**Cause:** SVG y goes down, math y goes up.

**Fix:** Always work in math/data coordinates and flip in the screen-space transform: `sy(y) = H - PAD - (...)` not `sy(y) = PAD + (...)`.

### 13.5 `cursor: grab` not applying

If you set `cursor-grab` on a circle inside an SVG with `cursor-crosshair`, the cursor may still show crosshair on hover. Use inline styles or ensure the handle's class wins.

### 13.6 Frame state not resetting when inputs change

**Symptom:** New dataset selected, but playback continues from the old frame index.

**Fix:** Always include this effect:
```tsx
useEffect(() => {
  setFrameIdx(0)
  setPlaying(false)
}, [frames])  // depend on the frames array reference
```

---

## 14. Roadmap

### Shipped (don't touch unless explicitly asked)

- **Chapter 18** — k-means clustering ✅
- **Chapter 6** — Gradient descent ✅
- **Chapter 7** — Logistic regression ✅

### Up next (recommended order)

**Chapter 8 — Multi-class classification.** The natural sequel to Chapter 7. Generalizes binary logistic regression to k classes via softmax. Three viz ideas:

1. **SoftmaxExplorer** — sigmoid's multi-class cousin. Show 3 input z-values; softmax outputs sum to 1. Drag the z-values, watch probabilities redistribute.
2. **MultiClassBoundary** — 3-class 2D dataset (e.g., three Gaussian blobs). Linear decision boundaries between every class pair form a Voronoi-like partition.
3. **OneVsRestVsSoftmax** — comparison of one-vs-rest and direct softmax on the same data. They give different boundaries; show why.

Pedagogical arc: §1 the problem (k classes), §2 one-vs-rest (the obvious extension), §3 limits of one-vs-rest (probabilities don't sum to 1), §4 the softmax fix, §5 categorical cross-entropy, §6 training (gradient is again `(p - y) x` — beautiful), §7 the decision regions form a Voronoi partition, §8 calibration with k classes, §9 complexity, §10 from scratch, §11 problems.

**Chapter 9 — Regularization.** L1 and L2. Geometric story of the constraint regions (L2 ball vs L1 diamond). Why L1 produces sparsity. Combine with logistic regression from Chapter 7. Viz ideas: L1 vs L2 path through the constraint region, regularization strength sweep showing coefficients shrinking, the bias-variance tradeoff visualized.

**Chapter 1 — What is machine learning?** The front door. Mostly prose; no need for as many widgets. Establishes the conceptual frame for the whole book. Sets up the "model + loss + optimizer" triad. Defines supervised vs unsupervised. Worth writing once Part II is solid because we'll have concrete examples to point to.

**Chapters 2–5 (Foundations + Linear regression).** Standard build-up. Linear regression (Ch 5) is the closed-form prequel to Chapter 6's iterative approach.

**Chapters 10–11 (Evaluation).** Train/val/test splits, cross-validation, metrics for classification and regression, calibration. Can reference back to Chapter 7 §8 (probability is information).

**Chapters 12–14 (Trees).** Decision trees, random forests, gradient boosting. Completely different visualizations needed — tree diagrams, recursive partitioning. Plan a new component pattern for tree rendering.

**Chapters 15–17 (Other classical).** SVMs, naive Bayes, feature engineering. SVMs share visual language with logistic regression (decision boundaries, margins).

**Chapters 18–20 (Unsupervised).** 18 done. 19 = hierarchical/density-based clustering. 20 = PCA + dimensionality reduction.

**Chapters 21–22 (Neural networks).** The crown of the book. 21 = MLP from scratch (chain rule, backprop). 22 = practical deep learning (Adam, regularization, training tricks). These will be the most ambitious chapters; plan for 4-5 widgets each.

### Other priorities

- **Landing page polish.** TOC could use better visual treatment of the part headers.
- **Reading-time estimates.** Show "~25 min" on each chapter card.
- **"Previous chapter" link** in chapter footer (currently only "Next chapter").
- **Print stylesheet** for chapters that read well on paper.

---

## 15. Tips for productive Claude Code sessions

### Read this first, every session

The voice of the book is specific. Spend 3 minutes reading `docs/VOICE-AND-STYLE.md` before drafting any new prose. If you start writing in "tutorial voice" the result will feel off — match the existing chapters or it won't ship.

### Read the most similar existing chapter before starting a new one

Going to write Chapter 8 (multi-class classification)? Read Chapter 7 (binary classification) end-to-end first. The structure transfers almost completely; deviation should be intentional.

### Reuse library code aggressively

Before writing a new `lib/foo.ts`, check `lib/gradient.ts`, `lib/logistic.ts`, `lib/contours.ts`. The patterns are deliberately general. If you find yourself writing a new optimizer, see if `runOptimizer` already handles your case.

### Components are throwaway-ish; libs are forever

Components are tied to a specific chapter and don't need to be reusable. Libs should be designed to last. Keep algorithms in libs and rendering in components.

### Verify the build before claiming you're done

```bash
npx next build
```

Always run this. TypeScript errors don't show in dev mode the way they do in build. Don't push without a clean build.

### Smoke-test the production output

```bash
npx next start -p 3000 &
sleep 5
curl -s http://localhost:3000/chapters/N-slug | grep katex-html | wc -l   # > 50?
curl -s http://localhost:3000/chapters/N-slug | grep -c "<details"        # = 5?
```

### Don't introduce new dependencies casually

The current dependency set is intentionally lean. Adding a library (e.g., d3, recharts) means new bundle weight and a maintenance liability. SVG by hand is almost always good enough.

### The author cares about the prose more than the code

The visualizations are flashy, but the writing is the book. When in doubt, spend the extra time on a paragraph rather than polishing a button hover. The voice — confident, direct, never condescending — is the most important thing.

### When uncertain about a content choice, prefer the option that compounds

The book builds. Chapter 7 builds on Chapter 6's GD. Chapter 8 will build on Chapter 7's logistic regression. Each chapter should leave the reader with new conceptual machinery that subsequent chapters use. If you're choosing between two ways to present an idea, pick the one that makes the *next* chapter easier to write.

---

*Last updated: end of Chapter 7 build session. Author: previous Claude conversation, working with Siddhant.*
