# Design plan: Chapter 13 — Random forests

> Detailed plan for Chapter 13, written before any code (per the recipe).
> The second chapter of **Part IV — Tree-based models**. Builds directly on
> Chapter 12: it takes the single tree's one great weakness — instability /
> high variance — and turns it into the engine of one of the most dependable
> models in machine learning. Foreshadowed by Chapter 12's closing line.

---

## Pedagogical role

A single decision tree is interpretable but *unstable*: Chapter 12 §8
showed that perturbing a few training points can grow a completely
different tree (high variance). A random forest accepts that instability
and exploits it — it grows hundreds of deliberately *different* trees and
averages their votes, and the averaging cancels the variance while
preserving the signal. The result is a near-drop-in, low-tuning model that
is hard to beat on tabular data.

The chapter introduces two sources of deliberate randomness (bootstrap
sampling and feature subsampling), the bias–variance argument for why
averaging helps, and the out-of-bag trick that hands you a free validation
set. It **reuses `lib/tree.ts` almost entirely** — a forest is just many
Chapter-12 trees — with one small extension: the tree builder must be able
to consider a random *subset* of features at each split.

It compounds on the whole book so far:
- Chapter 12 (trees) is the building block.
- Chapter 11 (cross-validation): out-of-bag error is a *free*
  cross-validation estimate, and Problem 5 makes that explicit.
- Chapter 7 §7 / Chapter 12: the forest's smooth boundary on moons is the
  next step in the long "linear models can't, and a single tree is blocky"
  thread.

It sets up Chapter 14 (gradient boosting): forests grow trees *in parallel*
and average; boosting grows them *in sequence*, each correcting the last.

---

## The conceptual arc (11 sections)

1. **The wisdom of crowds.** Sensory hook: a single person guessing
   jellybeans in a jar is wildly off, but the *average* of a thousand
   guesses is startlingly accurate — independent errors cancel, the shared
   signal survives. A single tree is an unstable guesser (Chapter 12); a
   forest is the crowd. Hook + promise.

2. **Why not just train many trees?** The central tension. Run the
   Chapter-12 algorithm a hundred times on the same data and you get the
   same tree a hundred times — averaging identical models buys nothing. A
   forest only works if its trees genuinely *differ*. The rest of the
   chapter is two ways to force that difference.

3. **Bagging: bootstrap aggregating.** The first source of randomness. Draw
   a *bootstrap sample* — n points sampled from the n training points *with
   replacement* — and train one tree on it. Repeat with fresh bootstrap
   samples. Each tree sees a slightly different dataset, so each grows a
   slightly different shape; classify by majority vote (regression: by
   average). → **Widget 1: BaggingBootstrap** (the original data, a
   bootstrap sample with repeats sized up and the ~37% never-drawn points
   hollowed out, and the single tree that sample grows — reshuffle to see
   the tree change).

4. **Why averaging reduces variance.** The bias–variance argument. Averaging
   $B$ independent estimators, each of variance $\sigma^2$, gives variance
   $\sigma^2 / B$ — averaging shrinks variance without raising bias. Real
   trees are not independent, though; with pairwise correlation $\rho$ the
   variance of the average is
   $\rho\,\sigma^2 + \frac{1-\rho}{B}\sigma^2$,
   which falls to $\rho\,\sigma^2$ as $B \to \infty$. Averaging alone hits a
   floor set by how *correlated* the trees are — which motivates the second
   trick.

5. **Feature subsampling: the "random" in random forest.** Bagged trees are
   still correlated, because they all seize on the same few strong features
   at the top. The fix: at each split, consider only a random subset of the
   features (the classic choice is $\sqrt{p}$). This decorrelates the trees —
   lowering $\rho$ and so the variance floor — at the cost of making each
   individual tree a little weaker. The combination of bootstrap rows and
   subsampled features *is* the random forest. → **Widget 2: ForestBoundary**
   (a single blocky tree boundary on the left, the forest's smooth
   vote-probability map on the right; a slider for the number of trees $B$
   and a toggle for feature subsampling).

6. **Out-of-bag error.** A bootstrap sample of size $n$ leaves out, on
   average, the fraction $(1 - 1/n)^n \to 1/e \approx 0.368$ of the points —
   about 37% are *out-of-bag* for each tree. Predict each point using only
   the trees that did *not* train on it, and you get an honest held-out
   estimate for free, no separate validation split required. Out-of-bag
   error closely tracks test error and is the forest's built-in answer to
   Chapter 11.

7. **What the forest buys.** The payoff, quantified. A single tree's
   boundary is blocky and jumps around with the data (high variance); the
   forest's boundary is smooth and stable, and its accuracy climbs with $B$
   and then plateaus — more trees never overfit, they only refine, with
   diminishing returns. Plus the practical notes that fit here: number of
   trees is safe to raise (unlike depth, adding trees cannot overfit) and
   training is embarrassingly parallel. → **Widget 3: VarianceCurve** (test
   and out-of-bag accuracy versus $B$, with the scattered accuracies of
   individual trees shown for contrast).

8. **Reading the black box: feature importance.** A forest trades the single
   tree's interpretability for accuracy — five hundred trees are no longer a
   readable flowchart. But it hands back one kind of insight: averaging, over
   the whole forest, the impurity decrease each feature is responsible for
   gives a *feature importance* ranking. On data with a few signal features
   buried among noise, the forest reliably scores the signal features high
   and the noise near zero — the same diagnosis the lasso gave in Chapter 9,
   from a completely different mechanism. → **Widget 4: FeatureImportance**
   (a forest trained on a Chapter-9-style signal-plus-noise dataset, with a
   bar chart of per-feature importances; toggle the number of trees and
   watch the ranking stabilise).

9. **Complexity.** Training is $B$ independent tree builds — $B$ times a
   single tree, and trivially parallel across trees. Prediction is
   $B \times O(\text{depth})$, still fast. Memory is $B$ trees. The constants
   are friendly enough that forests of hundreds of trees are routine.

10. **Implementing it yourself.** A random forest in ~35 lines of NumPy —
    bootstrap rows, a per-split random feature subset, and a majority vote —
    reusing the Chapter-12 tree wholesale.

11. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual** — Why does averaging many trees reduce variance but not
   bias? And why is bootstrap sampling alone not enough — what does feature
   subsampling add, in terms of the $\rho\sigma^2$ floor? (No code.)
2. **Bootstrap and out-of-bag** — implement bootstrap sampling and verify
   empirically that the out-of-bag fraction converges to $1/e \approx
   0.368$.
3. **The ensemble vote** — given a list of trained trees, predict by
   majority vote, and show the forest beats the average single tree.
4. **A random forest end to end** — bootstrap rows + a random feature per
   split + vote; compare a forest's accuracy to one tree's on held-out data.
5. **Out-of-bag as free cross-validation** — compute OOB accuracy as $B$
   grows, show it plateaus (more trees never hurt) and that it closely
   matches a held-out test score — a validation estimate for free. The
   compounding problem: Chapter 11 + Chapter 13.

---

## The four visualisations (detailed)

A new ensemble-rendering family that reuses Chapter 12's partition + tree
renderers. Widgets 1–3 share a 2-D `Point[]` dataset and the algorithms in
`lib/tree.ts` + a new `lib/forest.ts`; Widget 4 uses a higher-dimensional
signal-plus-noise dataset and renders a bar chart.

### Widget 1 — `BaggingBootstrap` (build first; the bootstrap concept)

- Two panels. **Left:** the original 2-D dataset; after a bootstrap is
  drawn, each point's radius scales with how many times it was sampled, and
  the ~37% never-drawn (out-of-bag) points are hollowed/greyed. **Right:**
  the single Chapter-12 tree grown on that bootstrap sample, shown as its
  rectangular partition.
- A **"New bootstrap"** button redraws the sample (seeded) — the left
  panel reshuffles which points are in/out, and the right panel's tree
  *visibly changes shape*.
- A live readout of the out-of-bag fraction for the current draw (hovering
  around 0.37).
- Pedagogical job: a bootstrap sample is a believable but different
  dataset; the tree it grows is different too; and roughly a third of the
  data sits out each time, ready to validate. The instability Chapter 12
  warned about is here recast as the raw material a forest will average
  away.

### Widget 2 — `ForestBoundary` (centrepiece; the vote)

- Two panels over the same data (default **moons**, with a toggle). **Left:**
  the decision boundary of a *single* tree — blocky, axis-aligned, twitchy.
  **Right:** the forest's **vote-probability map** — at each grid cell, the
  fraction of the $B$ trees voting for class 1, blended to a smooth heatmap.
- A **slider for $B$** (1 → 100): at $B = 1$ the right panel matches the
  blocky single tree; as $B$ grows the staircase artefacts average out and
  the boundary becomes smooth and soft, the probabilities graded rather than
  hard. A **feature-subsampling toggle** shows the decorrelation: with
  subsampling on, the forest smooths faster and generalises better.
- The forest is trained once (100 trees, memoised on dataset + toggle); the
  slider just aggregates the first $B$ trees' votes, so it stays responsive.
- Pedagogical job: *this is the random forest.* Many high-variance blocky
  trees, averaged, produce a low-variance smooth classifier — the wisdom of
  crowds made visible on a decision boundary.

### Widget 3 — `VarianceCurve` (the variance-reduction payoff)

- Accuracy versus the number of trees $B$ (1 → 100). The **forest's test
  accuracy** (teal) climbs and plateaus; the **out-of-bag accuracy**
  (dashed) tracks it closely — the free validation estimate. For contrast,
  the **individual trees' accuracies** are scattered as faint dots, visibly
  spread out and mostly *below* the forest line — the variance the averaging
  removes.
- Pedagogical job: more trees only ever help (diminishing returns, never
  overfitting — the opposite of tree depth in Chapter 12), the forest beats
  any single tree, and out-of-bag error is a cross-validation estimate you
  get for nothing. Reuses nothing from `crossval.ts` directly, but is the
  visual argument that OOB *is* cross-validation.

### Widget 4 — `FeatureImportance` (reading the black box)

- A forest trained on a **Chapter-9-style dataset**: a handful of signal
  features whose weights drive the label, plus several pure-noise features
  (reuses `makeSparseDataset` from `lib/regularisation.ts`, which already
  returns `{ x: number[]; y }`). This is the widget that needs $p > 2$
  features — the reason `lib/tree.ts` is generalised beyond 2-D.
- A horizontal **bar chart of per-feature importance**: for each feature,
  the total impurity decrease at the splits that used it, averaged across
  the forest and normalised to sum to one. Signal features (solid)
  tower over the noise features (faint); a small number-of-trees slider
  shows the ranking settling down as $B$ grows.
- Pedagogical job: a forest is a black box, but it can still tell you *which
  inputs mattered* — and on planted signal-vs-noise data it recovers the
  truth, echoing the lasso of Chapter 9 by an entirely different route
  (impurity, not a penalty). The standard caveat gets a sentence: impurity
  importance is biased toward high-cardinality features, which permutation
  importance fixes.

---

## Files to create

```
lib/
  tree.ts                       ← GENERALISE to p dimensions: Point.x
                                  becomes number[], feature becomes number,
                                  bestSplit loops over a feature list (default
                                  all). buildTree gains an optional rng +
                                  reads opts.maxFeatures to draw a random
                                  feature subset per node. This single change
                                  delivers BOTH feature subsampling and the
                                  p>2 data the importance widget needs. Ch 12
                                  passes 2-D data and is unaffected (verify
                                  its build).
  forest.ts                     ← NEW: bootstrap, trainForest, forestPredict,
                                  forestVoteProbs, forestAccuracy, oob*,
                                  featureImportances.
components/
  BaggingBootstrap.tsx          ← Widget 1
  ForestBoundary.tsx            ← Widget 2 (single tree vs forest vote map)
  VarianceCurve.tsx             ← Widget 3
  FeatureImportance.tsx         ← Widget 4 (signal-vs-noise bar chart)
app/chapters/
  13-random-forests/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/13-random-forests'`) and
`app/sitemap.ts`. **Proposed slug: `13-random-forests`** (matches the
landing title "Random forests").

Reused, not rewritten: the whole of `lib/tree.ts` (build/predict/partition
helpers, generators), `createRng`/`gauss` (`lib/rng.ts`), the class colour
palette. No new dependencies.

---

## Library design sketch

```ts
// ── lib/tree.ts — generalise to p dimensions + feature subsampling ───

export type Point = { x: number[]; y: number }   // was [number, number]
//   feature on a split node becomes `number` (was 0 | 1)

// bestSplit considers only the given feature indices (default: all p).
export function bestSplit(
  points: Point[],
  opts: TreeOptions,
  features?: number[],            // NEW; defaults to 0..p-1
): Split | null

export type TreeOptions = {
  numClasses: number
  numFeatures?: number            // NEW; inferred from data if omitted
  maxDepth: number
  minSamplesLeaf: number
  criterion?: Criterion
  maxFeatures?: number            // NEW: features to try per split (√p, etc.)
}
// If opts.maxFeatures is set and an rng supplied, buildTree draws a random
// maxFeatures-subset of the features at each node before calling bestSplit.
export function buildTree(points: Point[], opts: TreeOptions, depth?: number, rng?: () => number): TreeNode
```

```ts
// ── lib/forest.ts — new ──────────────────────────────────────────────
import { createRng } from './rng'
import { Point, TreeNode, TreeOptions, buildTree, predict } from './tree'

export type Forest = { trees: TreeNode[]; oob: number[][]; numClasses: number }
//   trees[b]      — the b-th tree
//   oob[b]        — indices of the original points NOT in tree b's bootstrap

/** Bootstrap sample of size n: indices drawn with replacement, the per-index
 *  draw counts, and the out-of-bag indices (those never drawn). */
export type Bootstrap = { idx: number[]; counts: number[]; oob: number[] }
export function bootstrap(n: number, seed: number): Bootstrap

export type ForestOptions = {
  numTrees: number
  numClasses: number
  maxDepth: number
  minSamplesLeaf: number
  maxFeatures?: number          // √p in general; 1 for the 2-D widgets
  seed: number
}

/** Train numTrees trees, each on its own bootstrap sample with per-split
 *  feature subsampling. Records each tree's out-of-bag indices. */
export function trainForest(points: Point[], opts: ForestOptions): Forest

/** Vote fractions across the first `nTrees` trees (default: all). */
export function forestVoteProbs(forest: Forest, x: [number, number], nTrees?: number): number[]
export function forestPredict(forest: Forest, x: [number, number], nTrees?: number): number
export function forestAccuracy(forest: Forest, points: Point[], nTrees?: number): number

/** Out-of-bag accuracy using the first `nTrees` trees: each point is
 *  predicted only by the trees whose bootstrap excluded it. */
export function oobAccuracy(forest: Forest, points: Point[], nTrees?: number): number

/** Per-feature importance: total impurity decrease (gain × node size) at the
 *  splits using each feature, averaged across the forest, normalised to sum
 *  to one. Length = numFeatures. */
export function featureImportances(forest: Forest, numFeatures: number): number[]
```

Notes:
- The forest is trained **once** per dataset/options; the widgets' $B$
  slider passes `nTrees` to the aggregation functions, so re-rendering is a
  cheap re-vote over already-built trees (no retraining on slider drag).
- `bootstrap` is deterministic given the seed; `trainForest` seeds each
  tree's bootstrap and per-node feature rng from `opts.seed + b`. No
  `Math.random` anywhere.
- For the 2-D widgets `maxFeatures = 1` (pick one of the two features per
  split) — the smallest non-trivial subsample, which is also the most
  visually decorrelating.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Class 0 / 1 / 2 | `#3c5a8c` / `#c7522a` / `#5d8a3a` | points; region + heatmap blends |
| In-bag points | full class colour | Widget 1 left panel |
| Out-of-bag points | hollow / `var(--ink-faint)` | Widget 1 left panel |
| Single-tree boundary | class fills @ low alpha | Widget 2 left (blocky) |
| Forest vote map | class-blended heatmap | Widget 2 right (smooth) |
| Forest / test curve | `var(--accent)` | Widget 3 |
| Out-of-bag curve | `var(--accent)` dashed | Widget 3 |
| Single-tree scatter | `var(--ink-muted)` faint dots | Widget 3 (the variance) |
| Importance bars | `var(--accent)` (signal) / `var(--ink-faint)` (noise) | Widget 4 |
| Highlights / current | `var(--accent)` | new region, markers |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure
numbering 13.1, 13.2, 13.3.

---

## Opening paragraph (draft, in voice)

> Ask one person to guess the number of jellybeans in a jar and they will
> be off — perhaps wildly. Ask a thousand people and average their guesses,
> and the average is often uncanny, closer than almost any single guesser.
> The errors, being roughly independent, cancel; the shared signal
> survives. This is the wisdom of crowds, and it is the whole idea behind
> the random forest. A single decision tree, as Chapter 12 showed, is
> accurate enough but wildly unstable — move a handful of training points
> and it grows a different shape. On its own that instability is a flaw. A
> forest turns it into fuel.
>
> The plan is to grow not one tree but hundreds, each deliberately
> different from the others, and let them vote. We will see the two tricks
> that force the trees apart — training each on a random resample of the
> data, and letting each split consider only a random handful of the
> features — and the bias–variance argument for why averaging so many
> unstable models produces a stable one. We will also meet a small piece of
> magic: because each tree is trained on a resample, the points it happened
> to leave out form a ready-made validation set, and a forest can grade
> itself for free.
>
> By the end you will have grown a forest from the trees of the last
> chapter, watched its blocky boundaries melt into a smooth one, and seen
> why adding more trees, unlike adding more depth, never makes things worse.

## Closing paragraph (draft, in voice)

> A random forest is the wisdom of crowds applied to decision trees: many
> unstable, blocky, individually mediocre models, decorrelated on purpose
> and averaged into one smooth, stable, and remarkably accurate whole. It
> asks almost nothing of you — little tuning, no feature scaling, a free
> validation estimate in the bargain — which is why it remains a first
> thing to reach for on tabular data decades after its invention. The price
> is interpretability: a single tree is a flowchart you can read, a forest
> of five hundred is a black box that simply works.
>
> A forest grows its trees in parallel and lets them vote as equals, each
> blind to the others. The next chapter takes the opposite tack. It grows
> trees one at a time, in sequence, each one trained specifically to fix the
> mistakes the others have made so far — a far more aggressive way to
> combine weak models, and the engine behind the algorithms that win
> competitions on tabular data.
>
> ---
>
> *Next: Chapter 14 — Gradient boosting.* Where a forest averages
> independent trees, boosting builds them in sequence, each correcting the
> last — and a chain of weak learners becomes a strong one.

---

## Expected scope

- Generalise `lib/tree.ts` to p dimensions (+ feature subsampling) and
  **re-verify Chapter 12's build/widgets** are unaffected; one new lib
  (`lib/forest.ts`, ~110 lines) + **four** components.
- One MDX file (~800 lines).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 details, 5 "Show solution", sitemap entry).
- Bundle size: ~11–13 kB / ~272 kB First Load JS, in line with Ch 12.
- **Performance note:** train the forest once per dataset (memoised); the
  $B$ slider only re-votes over already-built trees. Verify a 100-tree
  train on ~150 points stays well under a second before shipping.

Total estimated work: a touch larger than Chapter 12 — the tree.ts
generalisation plus four widgets — but heavy reuse of Chapter 12's code
keeps it to one focused session.

---

## Decisions (signed off)

1. **Slug** — `13-random-forests`. ✓
2. **Widget 2 default dataset** — **moons**: the single tree's jagged
   staircase melting into the forest's smooth curve is the chapter's
   headline image, and moons makes the contrast most dramatic. Overlapping
   blobs + XOR on the toggle.
3. **Four widgets** — bootstrap → forest boundary → variance curve →
   **feature importance**. The fourth pays off Chapter 9's lasso thread
   ("which features matter?") and is a headline reason forests are used in
   practice. It requires $p > 2$ features, so `lib/tree.ts` is generalised
   from hardcoded-2-D to p-dimensional — the same change feature subsampling
   needs anyway, and future-proof for Chapter 14. Chapter 12 passes 2-D data
   and is behaviourally unchanged; its build is re-verified after the
   refactor.
```
