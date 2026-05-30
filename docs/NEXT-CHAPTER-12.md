# Design plan: Chapter 12 — Decision trees

> Detailed plan for Chapter 12, written before any code (per the recipe).
> Opens **Part IV — Tree-based models**. The first chapter to abandon the
> linear/curve visual language entirely: its signature widget is a *tree
> diagram + recursive rectangular partition, drawn side by side and
> synchronised*. Directly foreshadowed by Chapter 11's closing line.

---

## Pedagogical role

Every model in Parts II–III drew a straight line — in the raw input space
(logistic regression, Ch 7–8) or a transformed one (polynomial ridge,
Ch 9). A decision tree does something categorically different: it carves
the feature space into axis-aligned rectangles by asking one yes-or-no
question at a time. It is the most *interpretable* model in the book — you
can read a tree off a page and follow its reasoning by hand — and the most
*unstable*, which is exactly the weakness Chapters 13–14 (random forests,
boosting) exist to fix.

This chapter introduces a **new component pattern** the rest of Part IV
will reuse:
- A **tree-diagram renderer** (nodes, branches, split conditions, leaves
  coloured by predicted class), with a simple recursive layout.
- A **recursive partition renderer** (the same tree drawn as nested
  axis-aligned rectangles over the 2-D data).
- The two **synchronised**, driven by the book's existing frame-history
  playback skeleton — but rendering a growing tree instead of a moving dot.

It compounds hard on Chapter 11: a tree's depth is a complexity knob, an
unconstrained tree overfits to 100% training accuracy, and **cross-
validation (Ch 11) is the tool that picks the depth.** Problem 5 calls
`cross_val_score` from Ch 11 directly. It also pays off the long-running
"linear models can't separate XOR / moons" thread (Ch 7 §7): a tree can.

---

## The conceptual arc (11 sections)

1. **A model that asks questions.** Sensory hook: emergency-room triage,
   a loan flowchart, a botanist's key — all reach a decision by a branching
   sequence of simple yes/no questions. That branching structure is a
   decision tree. Hook + promise.

2. **Which question, and in what order?** The central tension. A tree is
   just a sequence of threshold questions, but the number of possible trees
   is astronomical and finding the *optimal* tree is NP-hard. We need a
   tractable way to grow a good one.

3. **Greedy recursive splitting.** The key idea: don't search all trees.
   At each node, pick the single best feature-and-threshold split, then
   recurse on each side. Greedy and myopic, but fast and usually good.

4. **Measuring a split: impurity.** What makes a split "best"? A good split
   makes its children *purer* — closer to a single class. Gini impurity
   $G = 1 - \sum_c p_c^2$ (and entropy as the alternative); *information
   gain* = parent impurity − weighted child impurity. The algorithm sweeps
   every threshold on every feature and keeps the split of greatest gain.
   → **Widget 1: SplitFinder** (drag an axis-aligned split, watch child
   impurities and the information-gain-vs-threshold curve; toggle which
   feature to split on).

5. **Growing the tree: a partition into rectangles.** Apply the greedy
   split recursively until a stopping rule fires (pure node, max depth, too
   few samples). Each split is an axis-aligned cut, so the leaves tile the
   feature space with axis-aligned boxes — a tree approximates a curved
   boundary with a staircase, and, unlike any linear model, can carve apart
   non-linearly-separable data (the Ch 7 §7 payoff). →
   **Widget 2: TreeGrower** — the centrepiece dual view. Step through the
   greedy splits and watch the tree diagram deepen on one side while the
   feature-space partition subdivides in lockstep on the other. Default
   dataset **moons** (the recurring "linear models fail here" example —
   watching a tree conquer it pays off a five-chapter thread, and it grows
   cleanly with positive gain at every step), with **XOR/checkerboard** and
   **blobs** on the toggle.

6. **Overfitting and the depth knob.** Left unconstrained, a tree splits
   until every leaf holds one point — 100% training accuracy, memorised
   noise, jagged islands in the partition. Depth, `min_samples_leaf`, and
   cost-complexity pruning are the brakes, and the right setting is found
   by **cross-validation (Ch 11)**. → **Widget 3: DepthOverfit** — a
   depth slider; the partition fragments on the left while train accuracy
   climbs to 100% and held-out accuracy traces the familiar overfitting
   U on the right.

7. **Regression trees.** The same recursive partitioning, but the target is
   a number. Replace Gini with *variance reduction* and let each leaf
   predict the **mean** of its training targets — the tree becomes a
   piecewise-constant (step) function. → **Widget 4: RegressionTree** — a
   1-D scatter of the smooth Ch 9 curve; a depth slider grows the step
   function from a single flat line to an ever-finer staircase that chases
   the noise. The regression analogue of Widget 3's overfitting story, in
   one dimension.

8. **Practical concerns.** Interpretability (read the rules aloud).
   Instability / high variance (perturb a few points → a very different
   tree — the motivation for Ch 13). No feature scaling needed — splits are
   threshold-based and scale-invariant, the first model in the book for
   which standardisation is irrelevant. Native handling of mixed
   numeric/categorical features. Greedy myopia.

9. **Complexity.** Building: at each node, sort each feature's values to
   scan thresholds — about $O(n\,p\log n)$ per level, $O(n\,p\log^2 n)$ to
   grow a balanced tree. Prediction: $O(\text{depth}) \approx O(\log n)$.
   Memory $O(\text{nodes})$. Cheap to train, very cheap to predict.

10. **Implementing it yourself.** A recursive Gini tree builder + predict in
    ~35 lines of NumPy.

11. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual** — Why is a greedily grown tree generally *not* the
   globally optimal tree, and why does an unconstrained tree reach 100%
   training accuracy on any dataset with no two identical feature vectors?
   (No code.)
2. **Implement Gini impurity** from class counts; sanity-check pure vs
   uniform nodes.
3. **Implement the best-split finder** — sweep every threshold on every
   feature, return the (feature, threshold) of greatest information gain.
4. **Build a tree recursively and predict** — combine #2 and #3 into a
   depth-limited recursive builder; report training accuracy.
5. **Tune depth with cross-validation** — using Chapter 11's
   `cross_val_score`, sweep `max_depth`, show training accuracy marching to
   100% while CV accuracy peaks at a modest depth and then falls. The
   compounding problem: Ch 11 + Ch 12 in one.

---

## The four visualisations (detailed)

A genuinely new rendering family. Widgets 1–3 share one `Point[]` dataset
shape and the classification algorithms in `lib/tree.ts`; Widget 4 is the
1-D regression analogue.

### Widget 1 — `SplitFinder` (build first; the single-split concept)

- A 2-D scatter of two (or three) classes — colours from the Ch 8 palette.
- A **draggable axis-aligned split line**: a vertical cut $x_1 < t$ or a
  horizontal cut $x_2 < t$, with a **feature toggle** for which axis. The
  two regions shade faintly by their majority class.
- Live readouts: each child's class composition (small stacked bars), each
  child's **Gini impurity**, and the **information gain** of the current
  split. A small **gain-vs-threshold curve** beneath the scatter sweeps the
  whole axis and marks the maximising threshold.
- Pedagogical job: a good split drives child impurity down and gain up; the
  algorithm picks the gain peak — and the feature toggle shows it also
  chooses *which* feature. The simplest widget, and the conceptual atom of
  everything that follows.

### Widget 2 — `TreeGrower` (centrepiece; the dual view)

The new pattern. Two synchronised panels inside one figure:
- **Left — partition panel.** The 2-D data with the recursive
  axis-aligned partition: starting from the whole box, each split cuts the
  current region in two; leaves are filled with their majority-class
  colour. The region split at the current step is outlined in accent.
- **Right — tree-diagram panel.** The tree drawn top-down: internal nodes
  are boxes showing the split condition (`x₁ < 0.42`) in mono; leaves are
  class-coloured boxes; edges carry `yes`/`no`. The node added at the
  current step pulses in accent.
- **Playback** over the book's frame-history skeleton: the tree is grown
  *best-first* (each step splits the frontier leaf of greatest gain), so
  one frame = the tree after k splits. Play / Step / Reset, plus a
  **max-depth slider** and a **dataset toggle** (blobs · XOR/checkerboard ·
  moons). XOR is the headline: a tree shreds it in a few splits, where
  logistic regression (Ch 7) was helpless.
- Pedagogical job: make "greedy recursive partitioning" visible as a single
  process seen two ways — the abstract flowchart and the concrete geometry,
  moving together.

### Widget 3 — `DepthOverfit` (the overfitting / CV tie-in)

- A noisy 2-class dataset (two overlapping blobs, or moons + noise) where
  deep trees fit the noise.
- A **max-depth slider** (1–10). Left: the partition at that depth,
  visibly fragmenting into tiny single-point islands as depth grows.
  Right: **training accuracy** (grey, dashed) climbing to 100% and
  **held-out / cross-validated accuracy** (teal, with per-fold error bars)
  tracing the overfitting U — the same shape as Figure 11.1/11.3, now for
  tree depth. The CV-optimal depth is marked.
- Pedagogical job: an unconstrained tree memorises; depth is its complexity
  dial; cross-validation (Ch 11) finds the right setting. Reuses
  `kFoldSplit` from `lib/crossval.ts` — an explicit Part III callback.

### Widget 4 — `RegressionTree` (the 1-D regression analogue; a small widget)

- A 1-D scatter of the smooth Ch 9 curve (`makeNoisyScatter`/`trueCurve`
  from `lib/regularisation.ts` — reused, not regenerated): x along the
  axis, noisy y.
- A **max-depth slider** (0–8). The fitted regression tree is drawn as a
  **piecewise-constant step function** over the data: depth 0 is a single
  flat line at the global mean; each extra level halves regions and refines
  the steps; by high depth the staircase chases individual points. The true
  curve is shown dashed for reference.
- Readouts: number of leaves (pieces), train MSE, and a held-out MSE so the
  overfitting is quantified (the staircase's MSE bottoms out then rises).
- Pedagogical job: a regression tree predicts the **mean** of each leaf, so
  its output is a step function; depth controls how fine the steps are; too
  fine and it fits the noise. The regression mirror of Widget 3, and a
  bridge from the classification trees to the boosting of Chapter 14.

---

## Files to create

```
lib/
  tree.ts                       ← Point, TreeNode, gini/entropy, bestSplit,
                                  buildTree, buildTreeFrames (best-first
                                  playback), predict, accuracy, leaf/depth
                                  helpers, 2-D data generators (blobs, XOR,
                                  moons)
components/
  SplitFinder.tsx               ← Widget 1
  TreeGrower.tsx                ← Widget 2 (dual view: partition + tree)
  DepthOverfit.tsx              ← Widget 3
  RegressionTree.tsx            ← Widget 4 (1-D piecewise-constant fit)
app/chapters/
  12-decision-trees/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/12-decision-trees'`) and
`app/sitemap.ts`. **Proposed slug: `12-decision-trees`** (matches the
landing title "Decision trees").

Reused, not rewritten: `createRng`/`gauss` (`lib/rng.ts`); `kFoldSplit`/
`trainTestSplit` (`lib/crossval.ts`, for Widget 3 + Problem 5); the
three-class colour palette (Ch 8). The only new algorithm lib is
`tree.ts`. No new dependencies — tree layout and partition subdivision are
plain SVG done in the components.

---

## Library design sketch — `lib/tree.ts`

```ts
import { createRng, gauss } from './rng'

/** A 2-D labelled point. y is an integer class in {0,1,...,K-1}. */
export type Point = { x: [number, number]; y: number }

/** A node is either an internal split or a leaf. Both carry the class
 *  counts and sample size of the data that reached them, for rendering. */
export type TreeNode =
  | {
      kind: 'leaf'
      prediction: number      // majority class
      counts: number[]        // per-class counts at this node
      n: number
      depth: number
    }
  | {
      kind: 'split'
      feature: 0 | 1          // which axis the cut is on
      threshold: number       // cut location; go left if x[feature] < threshold
      gain: number            // information gain of this split
      left: TreeNode
      right: TreeNode
      counts: number[]
      n: number
      depth: number
    }

export type TreeOptions = {
  numClasses: number
  maxDepth: number
  minSamplesLeaf: number
  criterion?: 'gini' | 'entropy'  // default 'gini'
}

// ── Impurity ────────────────────────────────────────────────────────
export function gini(counts: number[]): number      // 1 - Σ p_c²
export function entropy(counts: number[]): number    // -Σ p_c log p_c

// ── Splitting ───────────────────────────────────────────────────────
/** Sweep every candidate threshold (midpoints of sorted unique values) on
 *  both features; return the split of greatest information gain, or null if
 *  no split beats the parent (or would violate minSamplesLeaf). */
export function bestSplit(
  points: Point[],
  opts: TreeOptions,
): { feature: 0 | 1; threshold: number; gain: number } | null

// ── Building ────────────────────────────────────────────────────────
/** Greedy recursive build to the stopping rules in opts. */
export function buildTree(points: Point[], opts: TreeOptions): TreeNode

/** Best-first growth for playback: returns the tree after 0, 1, 2, …,
 *  maxSplits splits, each step expanding the frontier leaf of highest gain.
 *  `lastPath` is the root→node path of the leaf just split (for highlight). */
export type TreeFrame = { root: TreeNode; nSplits: number; lastPath: number[] }
export function buildTreeFrames(
  points: Point[],
  opts: TreeOptions,
  maxSplits: number,
): TreeFrame[]

// ── Use ─────────────────────────────────────────────────────────────
export function predict(node: TreeNode, x: [number, number]): number
export function accuracy(node: TreeNode, points: Point[]): number
export function countLeaves(node: TreeNode): number
export function treeDepthOf(node: TreeNode): number

// ── Regression trees (Widget 4) — 1-D x → continuous y ──────────────
export type RegPoint = { x: number; y: number }
export type RegNode =
  | { kind: 'leaf'; value: number; n: number; depth: number }
  | { kind: 'split'; threshold: number; left: RegNode; right: RegNode; n: number; depth: number }
/** Greedy build splitting on variance reduction; each leaf predicts the
 *  mean target of the points that reach it (a step function). */
export function buildRegressionTree(points: RegPoint[], maxDepth: number, minLeaf: number): RegNode
export function predictReg(node: RegNode, x: number): number
/** Leaf boundaries + values, left→right, for drawing the step function. */
export function regSteps(node: RegNode, xMin: number, xMax: number): { x0: number; x1: number; value: number }[]

// ── 2-D datasets (math coords centred near origin, like the Ch 7/8 gens) ──
export function makeBlobs2(n: number, seed: number, k?: number): Point[]   // k Gaussian blobs
export function makeXOR(n: number, seed: number): Point[]                  // 2-class checkerboard
export function makeMoons2(n: number, seed: number): Point[]               // 2 interleaved arcs
// (Widget 4 reuses makeNoisyScatter / trueCurve from lib/regularisation.ts.)
```

Notes:
- `bestSplit` evaluates candidate thresholds at midpoints of sorted unique
  values per feature — standard, and exact for axis-aligned trees.
- `buildTreeFrames` uses a frontier priority queue keyed by gain, so each
  playback step is the single most valuable split remaining — the order in
  which the partition visibly improves fastest.
- Layout of the tree diagram and subdivision of the partition box are
  **component-side** (they are pixel concerns); the lib returns only the
  abstract `TreeNode` structure. This keeps `tree.ts` reusable for Ch 13–14.
- Everything is deterministic given the seed (`createRng`, no `Math.random`).

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Class 0 | `#3c5a8c` (blue) | points + region fill (low alpha) |
| Class 1 | `#c7522a` (orange) | points + region fill |
| Class 2 | `#5d8a3a` (olive) | points + region fill (3-class blobs) |
| Internal tree node | `var(--paper-soft)` + `var(--rule)` border | box with split condition (mono) |
| Leaf node | class colour (low alpha) | box, predicted class |
| Edges / partition cuts | `var(--rule)` | branches, region boundaries |
| Current split / node | `var(--accent)` | the node + region added this step; gain-curve peak; CV-best depth |
| Training curve | `var(--ink-muted)` dashed | the misleading curve (Widget 3) |
| Held-out / CV curve | `var(--accent)` | the honest curve + error bars (Widget 3) |

Fonts: `font-sans` for UI/labels, `font-mono` for split conditions and
numeric readouts. Figure numbering 12.1, 12.2, 12.3.

---

## Opening paragraph (draft, in voice)

> A doctor in a busy emergency room does not fit a regression. She asks one
> question — is the patient short of breath? — and, depending on the answer,
> asks another, and then another, narrowing toward a diagnosis one yes-or-no
> at a time. A bank clerk weighing a loan, a botanist keying out a
> wildflower, the troubleshooting flowchart taped inside a photocopier: all
> of them reach a decision the same way, by walking down a branching
> sequence of simple questions. That branching structure is a *decision
> tree*, and it is among the most intuitive models in machine learning — you
> can print one on a single page and follow its reasoning by hand, no
> arithmetic required.
>
> This chapter builds decision trees from the ground up. We will see how a
> tree decides which question to ask first, how it grows itself greedily one
> split at a time, and how those questions carve the feature space into a
> patchwork of rectangles — a shape no linear model of the last six chapters
> could ever draw. We will also meet the tree's great weakness, the one the
> next two chapters exist to repair: left unchecked, a tree keeps asking
> questions until it has memorised the training set whole.
>
> By the end you will have grown a tree from scratch, watched it partition a
> plane, and used the cross-validation of Chapter 11 to stop it before it
> overfits.

## Closing paragraph (draft, in voice)

> A decision tree is the most legible model in this book: a flowchart you
> can read aloud. It is also one of the most temperamental. Because every
> split is chosen greedily and the whole structure hangs from the first cut,
> moving a handful of training points can grow an entirely different tree —
> high variance, in the language of Chapter 11. A single tree is
> interpretable, but rarely the most accurate thing you can build.
>
> The next chapter turns that weakness into a strength with an idea that
> sounds too simple to work: if one tree is unstable, grow hundreds of them,
> each on a slightly different sample of the data and the features, and let
> them vote. The averaging cancels the variance, and the resulting *random
> forest* is one of the most dependable off-the-shelf models in all of
> machine learning.
>
> ---
>
> *Next: Chapter 13 — Random forests.* Why a crowd of imperfect,
> decorrelated trees beats any single carefully-pruned one.

---

## Expected scope

- One new lib module (`lib/tree.ts`, ~180 lines — classification +
  regression trees, both recursive) + **four** components, one of which
  (`TreeGrower`) is the most involved widget in the book so far (tree
  layout + partition subdivision + playback).
- One MDX file (~800 lines).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 details, 5 "Show solution", sitemap entry).
- Bundle size: ~12–14 kB / ~272 kB First Load JS (heavier than Ch 7–11
  because of the tree-layout renderer + a fourth widget), still well under
  the 400 kB watch-line.

Total estimated work: larger than Chapters 8/10/11 — a new render pattern
and four widgets — but the algorithms are cheap and the arc is clear.

---

## Decisions (signed off)

1. **Slug** — `12-decision-trees`. ✓
2. **Widget 2 headline dataset** — **moons** by default (XOR/checkerboard +
   blobs on the toggle): it pays off the five-chapter "linear models fail
   on moons" thread, grows cleanly with positive gain at every greedy step,
   and teaches the staircase-approximation geometry as it builds. Symmetric
   XOR was rejected as the *default* because its zero-gain first split makes
   greedy growth look arbitrary — but it stays on the toggle for the
   dramatic interaction-effect exploration.
3. **Impurity criterion** — **lead with Gini** ($1 - \sum p_c^2$): no logs,
   the intuitive "chance two random points in this node disagree" reading,
   the clean 2-class parabola $2p(1-p)$, sklearn's default, and easiest to
   compute by hand in the problems. Entropy / "information gain" gets a
   one-sentence mention as the near-equivalent. Widgets compute Gini.
4. **Regression trees** — their own short **§7 + Widget 4**
   (`RegressionTree`), with the classification spine folding the old
   "geometry" section into §5 to keep the landmark 11-section shape.
```
