# Chapter Recipe

A complete, opinionated recipe for writing a new chapter. Follow these steps in order. Each step references files or sections in `HANDOFF.md` for details.

---

## Before you start

- [ ] Read `docs/VOICE-AND-STYLE.md` (15-min read)
- [ ] Read the most similar existing chapter end-to-end. For a linear-model chapter, read `app/chapters/7-logistic-regression/page.mdx`. For an unsupervised chapter, read `app/chapters/18-k-means/page.mdx`. For anything optimization-flavored, read `app/chapters/6-gradient-descent/page.mdx`.
- [ ] Decide on the **three visualization concepts** for the chapter. The third is usually the "main playback widget" that shows the algorithm running. The first two support it.
- [ ] Decide on the **five problem prompts**, in order of difficulty. The last problem should compound — apply the chapter's algorithm to something the reader couldn't have done before this chapter.

---

## Step 1 — Sketch the chapter outline

Write the 11 section headers in a scratch file with one-sentence summaries of each. Don't start writing yet. Verify the arc has the standard shape:

1. Conceptual setup (~250 words)
2. Naive approach fails / central tension (~400 words)
3. Key new idea (~400 words, often with widget 1)
4. Model / algorithm definition (~400 words, often with widget 2)
5. Loss / criterion (~500 words — derivation lives here)
6. Training / optimization (~400 words, with the main widget)
7. Geometric / structural consequence (~350 words)
8. Practical concerns (~400 words)
9. Complexity (~200 words)
10. Implementation from scratch (~150 words + 25-30 lines code)
11. Problems (5 problems, varied difficulty)

Total target: ~3,500 words of prose.

---

## Step 2 — Build the library code first

If the chapter introduces a new algorithm, write `lib/<algorithm>.ts` before any components or prose.

The lib should expose:
- A pure `runAlgorithm(input, opts)` function that returns `Frame[]` for playback
- Type definitions for the model state, frame, and options
- Any helper data generators specific to the chapter

Patterns to match:
- `lib/gradient.ts` — for iterative optimization with a frame-history return
- `lib/logistic.ts` — for ML algorithms with model + training + data generators
- `lib/kmeans.ts` — for assign/update style algorithms

Test the lib in isolation by writing a small smoke test or just running it in a `console.log` script. Don't move on until the algorithm is producing sensible output.

---

## Step 3 — Build the three components

In order of complexity, simplest first:

1. **Concept widget** (e.g., SigmoidExplorer) — single curve / static visualization with one interactive control. ~150-200 lines.
2. **Interactive playground** (e.g., DecisionBoundary) — multi-element scene with draggable handles and live updates. ~250-350 lines.
3. **Playback widget** (e.g., TrainingDynamics) — runs the algorithm with full playback controls. ~350-400 lines.

Each component follows the patterns in `HANDOFF.md` §6:
- `'use client'` directive at the top
- `useState` for parameters
- `useMemo` for expensive precomputation
- `useEffect` for playback animation and frame-reset on input change
- SVG with `viewBox`, coordinate transforms `sx`/`sy`
- Controls row → SVG container → playback row → stats row → figcaption

**Copy the structure of the existing component most similar to what you're building.** Don't start from scratch.

---

## Step 4 — Write the MDX

Now write the prose, with widgets embedded. Use `app/chapters/7-logistic-regression/page.mdx` as a template:

```mdx
import Widget1 from '@/components/Widget1'
import Widget2 from '@/components/Widget2'
import Widget3 from '@/components/Widget3'
import PyodideEditor from '@/components/PyodideEditor'

export const metadata = {
  title: '<Chapter title>',
  description: '<One-sentence description that will appear in Slack/Twitter previews — make it specific and slightly poetic>',
  openGraph: {
    title: '<Chapter title>',
    description: '<Same or similar description>',
    type: 'article',
  },
}

<p className="font-sans text-xs uppercase tracking-[0.18em] text-ink-muted mb-3">
  Chapter N · Part X — <Part name>
</p>

# <Chapter title>

*<One-line italic subtitle. Something the reader can see in their head.>*

<3 opening paragraphs>

## 1. <Section title>

<Prose. Embed widget if applicable.>

[... §2 through §10 ...]

## 10. Implementing it yourself

<Brief lead-in.>

<PyodideEditor initialCode={`<25-30 lines of from-scratch NumPy>`} />

## 11. Problems

### Problem 1 — <Title>

<Problem statement.>

[Starter code if applicable]

<details>
  <summary>Show solution</summary>

[Solution code]

[Brief explanation.]

</details>

[... Problems 2-5 ...]

---

*Next: Chapter N+1 — <Next title>.* <One-sentence hook.>
```

**Match the voice exactly.** Re-read `docs/VOICE-AND-STYLE.md` after writing each section. If a passage feels off, it probably is.

---

## Step 5 — Wire into navigation

### Landing page

Open `app/page.tsx`. Find the chapter entry for the chapter you wrote. Add the `href` field:

```ts
{ num: N, title: '<Title>', blurb: '<...>', href: '/chapters/<N>-<slug>' },
```

If the chapter doesn't have an entry yet, copy a neighboring chapter's entry as a template.

### Sitemap

Open `app/sitemap.ts`. Add the new chapter URL:

```ts
{
  url: `${base}/chapters/<N>-<slug>`,
  lastModified: now,
  changeFrequency: 'weekly',
  priority: 0.8,
},
```

---

## Step 6 — Build and smoke-test

```bash
npx next build
```

Verify the output shows the new chapter route with a reasonable size (3-component chapters land at ~6-8 kB / ~265 kB First Load JS).

If build fails:
- TypeScript errors → fix the types, never use `any`
- MDX parse errors → check for stray template-literal backticks or unclosed JSX
- Missing imports → verify all component imports at the top of the MDX

Then smoke-test the running production build:

```bash
npx next start -p 3000 &
sleep 5

# 200 status
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000/chapters/N-slug

# KaTeX rendering
curl -s http://localhost:3000/chapters/N-slug | grep -c "katex-html"
# expect > 50 for a chapter with math

# Problem structure
curl -s http://localhost:3000/chapters/N-slug | python3 -c "
import sys, re
html = sys.stdin.read()
print('Problems:', len(re.findall(r'<h3[^>]*>Problem \d', html)))
print('Details:', html.count('<details'))
print('Solutions:', len(re.findall(r'<summary[^>]*>Show solution</summary>', html)))
"
# expect: Problems: 5, Details: 5, Solutions: 5

# Sitemap
curl -s http://localhost:3000/sitemap.xml | grep <slug>

# Kill server
pkill -f "next start"
```

Also check in a browser:
- [ ] Landing page link works
- [ ] All three widgets render without console errors
- [ ] Sliders update the visualization smoothly
- [ ] Playback works (play / pause / step / scrub)
- [ ] PyodideEditor in §10 runs successfully (numpy works)
- [ ] Problems show their statement and starter code visible up-front
- [ ] Clicking "Show solution" reveals the solution PyodideEditor
- [ ] Math (KaTeX) renders correctly throughout
- [ ] "Table of contents" back-link returns to landing
- [ ] Dark mode looks acceptable

---

## Step 7 — Deploy

```bash
git add -A
git commit -m "Chapter N: <Title>"
git push
```

Vercel auto-deploys. The new chapter URL becomes live in ~2 minutes:

```
https://ml-book-seven.vercel.app/chapters/N-slug
```

---

## Common patterns to reach for

### Need a new probability density visualization?

Look at `DecisionBoundary.tsx`'s heatmap rendering (40×30 grid of `<rect>` with rgba fills, blue for class 0, orange for class 1).

### Need a contour plot?

`computeContours` from `lib/contours.ts` plus the rendering pattern in `GradientDescent2D.tsx`.

### Need a draggable handle?

Pointer-capture pattern in `DecisionBoundary.tsx` (search for `setPointerCapture`).

### Need a 1D function curve?

`GradientDescent1D.tsx` builds a path string. Sample 200 points, emit `M x y L x y L x y ...`.

### Need a loss-over-iterations chart?

The right panel of `TrainingDynamics.tsx`. Ghosted full curve behind the progressed-so-far curve.

### Need to train an ML model with playback?

`trainLogistic` from `lib/logistic.ts` returns `LRFrame[]` with model + loss + accuracy per step.

### Need to run an optimization with playback?

`runOptimizer` from `lib/gradient.ts` returns `GDFrame[]` with position + gradient + loss per step. Handles vanilla GD, momentum, SGD, or any combination via options.

---

## When you're stuck

- **The voice feels off.** Re-read `docs/VOICE-AND-STYLE.md`. Then re-read the most similar shipped chapter. Then rewrite the problem passage.
- **The widget feels generic.** What does this widget teach that text alone couldn't? If you can't answer in one sentence, the widget design is wrong.
- **The math is getting messy.** Step back and ask: what is the geometric/intuitive picture? Write that first, then the formal version.
- **The chapter is too long.** Compress §1-§3 (setup) ruthlessly. Don't add caveats in §7-§9; they should be terse.
- **The chapter is too short.** §5 (loss derivation) and §7 (geometric consequence) are usually under-developed when chapters feel thin. Add a worked example.
