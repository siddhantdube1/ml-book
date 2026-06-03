# Design plan: Chapter 14 — Gradient boosting

> Detailed plan for Chapter 14, written before any code (per the recipe).
> Closes **Part IV — Tree-based models**. The deliberate opposite of Chapter
> 13: where a forest grows trees in parallel and averages away *variance*,
> boosting grows them in sequence, each correcting the last, and drives down
> *bias*. Reuses Chapter 12's regression tree as the weak learner.

---

## Pedagogical role

Chapter 13 built an ensemble by *parallel averaging*: many deep,
low-bias/high-variance trees, decorrelated and averaged to cancel variance.
Chapter 14 builds the other kind of ensemble — *sequential correction*: many
shallow, high-bias/low-variance trees, each trained on the errors the
ensemble has made so far, summed to cancel bias. The two chapters are the
yin and yang of tree ensembles, and putting them back to back is the point.

Gradient boosting is also, in its modern incarnations (XGBoost, LightGBM,
CatBoost), the single most effective family of models on tabular data — the
thing that wins competitions and quietly runs in production everywhere. So
the chapter is both the conceptual close of Part IV and the practical apex
of the whole tree arc.

It compounds on the book:
- Chapter 12's *regression tree* is the weak learner, reused directly.
- Chapter 6 (gradient descent): boosting *is* gradient descent — in function
  space rather than parameter space — and saying so is the chapter's central
  reframe.
- Chapter 11 (cross-validation / early stopping): unlike a forest, a boosted
  model overfits, so the number of trees is a hyperparameter tuned by a
  validation curve.
- Chapter 13 (forests): the explicit bias-vs-variance, sequential-vs-parallel
  contrast.

It sets up Part V (Chapter 15, support vector machines): a return from
ensembles of weak rules to a single, optimally-placed boundary.

---

## The conceptual arc (11 sections)

1. **Learning from your mistakes.** Sensory hook: a student who, before each
   exam, does not reread the whole syllabus but studies precisely the
   questions they got wrong last time — correcting specific weaknesses, one
   at a time. Boosting is that idea as an algorithm, and the opposite of the
   forest's parallel crowd (Chapter 13). Hook + promise.

2. **The additive model.** The plan: build the prediction as a *sum* of
   small corrections, $F_M(x) = F_0(x) + \sum_{m=1}^M h_m(x)$. Start with the
   simplest possible prediction (a constant), then repeatedly add a function
   that fixes the current errors. The model grows one term at a time, never
   revising the terms already placed.

3. **Fitting the residuals.** The intuitive core, for regression. Start with
   $F_0 = \bar{y}$, the mean. Compute the *residuals* $r_i = y_i - F(x_i)$ —
   how wrong the ensemble is on each point. Fit a small regression tree to
   *predict those residuals*, and add it to the ensemble. The residuals
   shrink; recompute and repeat. Each tree patches the leftover error of the
   ones before it. → **Widget 1: ResidualBoosting** (a 1-D regression
   problem: the ensemble's prediction climbing toward the data on top, the
   shrinking residuals with the next tree being fit to them below; step
   through the rounds).

4. **Boosting is gradient descent in function space.** The reframe that
   names the algorithm. For squared loss $L = \frac12(y - F)^2$, the residual
   $y - F$ is exactly $-\partial L / \partial F$ — the negative gradient. So
   "fit a tree to the residuals" *is* "take a gradient step", except the step
   is taken in the space of *functions* $F$ rather than a vector of
   parameters. For other losses (logistic for classification, absolute error
   for robust regression) you fit the tree to the negative gradient — the
   *pseudo-residuals* — and the same machinery boosts any differentiable
   loss. This is the "gradient" in gradient boosting, and the tie back to
   Chapter 6.

5. **Shrinkage: the learning rate.** A subtlety that matters enormously in
   practice. Instead of adding each tree at full strength, scale it by a
   small *learning rate* $\nu$ (typically 0.1): $F \leftarrow F + \nu\,h_m$.
   Shrinking each step means more trees are needed, but the ensemble
   generalises better — many small steps beat a few large ones. The
   $\nu$-versus-number-of-trees trade-off is the central tuning knob. →
   **Widget 2: LearningRate** (the 1-D fit at a fixed budget of trees, with a
   $\nu$ slider: too small underfits, just right tracks the curve, too large
   overshoots into a jagged overfit).

6. **Weak learners, and the contrast with forests.** Boosting uses
   *shallow* trees — stumps, or trees of depth two or three — each a *weak
   learner*: high bias, low variance, barely better than guessing on its own.
   This is the exact opposite of a random forest's deep trees, and so is the
   mechanism: a forest *averages* low-bias/high-variance trees to kill
   variance; boosting *sums* high-bias/low-variance trees to kill bias. Same
   building block, opposite philosophies. It is also why a forest is
   embarrassingly parallel and boosting is stubbornly sequential.

7. **Boosting can overfit: early stopping.** The price of all that
   error-chasing. Add too many trees and the ensemble starts fitting the
   noise — training error keeps falling while held-out error bottoms out and
   turns back up. Unlike a forest, where more trees never hurt (Chapter 13),
   in boosting the number of trees is a genuine hyperparameter, and the
   standard recipe is *early stopping*: watch a validation curve and stop at
   its minimum. → **Widget 3: OverfitEarlyStop** (training and held-out error
   versus the number of trees, the test-error U, the early-stopping point
   marked; a $\nu$ slider shows smaller rates reaching lower minima later).
   The Chapter 11 tie and the sharp Chapter 13 contrast.

8. **Classification, stochastic boosting, and the libraries that won.**
   Three loose ends. *Classification*: model the log-odds additively, use
   logistic loss, and the pseudo-residual is the clean $y_i - p_i$ —
   probability error, just like Chapters 7–8. *Stochastic boosting*: fit each
   tree on a random subsample of the rows (and features), which both speeds
   training and regularises, borrowing the forest's trick. *The libraries*:
   XGBoost, LightGBM, and CatBoost are gradient boosting with second-order
   (Newton) steps, clever regularisation, and histogram-binned splits — the
   reason "just use gradient boosting" is the standard advice for tabular
   data. → **Widget 4: BoostingBoundary** (boosting a 2-D classifier on
   moons: the decision boundary sharpening round by round, set beside the
   memory of Chapter 13's smooth forest — sequential focus versus parallel
   averaging, drawn).

9. **Complexity.** Training is inherently *sequential* — tree $m$ needs the
   residuals from trees $1..m-1$ — so unlike a forest it does not parallelise
   across trees (though each individual tree build does). $M$ trees of depth
   $d$ on $n$ points: roughly $O(M \, n \, d)$ with the histogram trick, more
   without. Prediction is $M \times O(d)$, the same as a forest of the same
   size. Memory is $M$ shallow trees — far smaller than a forest of deep
   ones.

10. **Implementing it yourself.** Gradient boosting for regression in ~30
    lines, reusing Chapter 12's regression tree: initialise at the mean, then
    loop — residuals, fit a shallow tree, add $\nu$ times it.

11. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual** — A forest *averages* deep trees and boosting *sums*
   shallow ones. Explain how this maps onto bias and variance, and why it
   follows that more trees never hurt a forest but can wreck a boosted model.
   (No code.)
2. **One round of residual fitting** — start from the mean, compute
   residuals, fit a single shallow tree to them, and show the residuals
   shrink (training error drops).
3. **Gradient boosting from scratch** — the full regression loop (init,
   residuals, shallow tree, shrink, add) on the Chapter-9 curve; plot the fit
   improving with rounds.
4. **Learning rate and early stopping** — for two learning rates, plot
   held-out error versus the number of trees, and report each one's
   early-stopping round (the test-error minimum). Connects to Chapter 11.
5. **Boosting versus a forest** — train both on the same noisy dataset and
   compare: the forest plateaus and is safe to over-build; the boosted model
   peaks then declines and must be stopped. The Part IV capstone.

---

## The four visualisations (detailed)

Three share a 1-D regression dataset (the smooth Chapter-9 curve), reused so
the running example carries across the chapter; the fourth is a 2-D
classifier. All reuse `lib/tree.ts`'s regression tree via a new
`lib/boosting.ts`.

### Widget 1 — `ResidualBoosting` (the core; build first)

- Two stacked panels over the 1-D curve. **Top:** the data (faint dots), the
  true curve (dashed), and the ensemble's current prediction $F_m$ (a teal
  staircase) — which starts as a flat line at the mean and climbs toward the
  data as rounds are added. **Bottom:** the current *residuals* $y_i - F_m(x_i)$
  as a scatter about zero, with the *next* weak tree's fit drawn through them.
- **Playback** over the boosting rounds (Play / Step / Reset), plus a small
  weak-learner-depth control. Each step adds one tree: the top prediction
  gains detail, the bottom residuals shrink toward zero and re-centre, and
  the new tree visibly tracks whatever structure is left in the residuals.
- Pedagogical job: *this is boosting.* Each tree is trained on the leftover
  error, and the sum of many small corrections converges on the signal —
  seen as the residual cloud collapsing onto the zero line.

### Widget 2 — `LearningRate` (shrinkage)

- The same 1-D curve, the ensemble fit at a **fixed** number of trees (say
  60), with a **$\nu$ slider** (0.02 → 1). At tiny $\nu$ the fit has barely
  left the mean — sixty timid steps go nowhere (underfit, high bias); around
  $\nu = 0.1$ it tracks the curve cleanly; at $\nu \to 1$ each tree slams in
  at full strength and the fit turns jagged, chasing noise (overfit).
- A readout of train and held-out error makes the trade-off quantitative.
- Pedagogical job: shrinkage is the boosting regulariser. Small steps,
  patiently summed, generalise; large steps overshoot. It is the learning
  rate of Chapter 6, now governing an ensemble.

### Widget 3 — `OverfitEarlyStop` (early stopping; the Ch 11 + Ch 13 tie)

- Training error (grey, dashed) and held-out error (teal) versus the number
  of trees, both falling at first; training error keeps sinking toward zero
  while held-out error bottoms out and **turns back up** — the boosting
  overfitting U. The minimum (the early-stopping round) is marked. A $\nu$
  slider shifts the curve: smaller rates reach a lower minimum, but later.
- Pedagogical job: more trees is not free in boosting — there is an optimal
  number, found by a validation curve exactly as in Chapter 11, and stopping
  there is *early stopping*. The contrast with Chapter 13's forever-safe
  forest curve is the explicit lesson.

### Widget 4 — `BoostingBoundary` (classification; the forest contrast)

- A 2-D moons classifier boosted round by round. A grid heatmap of the
  ensemble's predicted probability, with a **rounds slider**: at one tree the
  boundary is a single crude stump-cut; as rounds accumulate it sharpens and
  bends to follow the two arcs, the probabilities hardening from the centre
  outward. A weak-learner-depth control sets how strong each tree is.
- Pedagogical job: boosting works for classification too (pseudo-residuals
  $y - p$), and its boundary is built by *sequential focus* — set this beside
  the memory of Chapter 13's smooth, parallel-averaged forest boundary on the
  same moons, and the two ensemble philosophies are visible side by side.

---

## Files to create

```
lib/
  tree.ts                       ← EXTEND the regression tree to p dimensions:
                                  RegPoint.x becomes number[], RegNode.split
                                  gains a feature index, buildRegressionTree
                                  splits on the best feature (variance
                                  reduction) across all dims, predictReg takes
                                  number[]. (Classification tree already
                                  p-dim from Ch 13.) regSteps stays a 1-D
                                  render helper. Update Ch 12's RegressionTree
                                  widget (1-D → x:[v]) and re-verify its build.
  boosting.ts                   ← NEW: GBRegressor / GBClassifier, train,
                                  staged predict, residuals, pseudo-residuals.
components/
  ResidualBoosting.tsx          ← Widget 1
  LearningRate.tsx              ← Widget 2
  OverfitEarlyStop.tsx          ← Widget 3
  BoostingBoundary.tsx          ← Widget 4
app/chapters/
  14-gradient-boosting/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/14-gradient-boosting'`) and
`app/sitemap.ts`. **Proposed slug: `14-gradient-boosting`** (matches the
landing title "Gradient boosting").

Reused: the regression tree of `lib/tree.ts` (the weak learner),
`makeNoisyScatter`/`trueCurve` (`lib/regularisation.ts`, the 1-D curve),
`makeMoons2` (`lib/tree.ts`, the 2-D classifier), `trainTestSplit`
(`lib/crossval.ts`), the class palette. No new dependencies.

---

## Library design sketch

```ts
// ── lib/tree.ts — generalise the regression tree to p dimensions ─────
export type RegPoint = { x: number[]; y: number }          // was { x: number }
export type RegNode =
  | { kind: 'leaf'; value: number; n: number; depth: number }
  | { kind: 'split'; feature: number; threshold: number; left: RegNode; right: RegNode; n: number; depth: number }
// buildRegressionTree splits on the best (feature, threshold) by variance
// reduction across all features; predictReg walks number[].
export function buildRegressionTree(points: RegPoint[], maxDepth: number, minLeaf: number): RegNode
export function predictReg(node: RegNode, x: number[]): number
// regSteps stays 1-D (assumes a single feature) — used only by the Ch 12 widget.
```

```ts
// ── lib/boosting.ts — new ────────────────────────────────────────────
import { RegPoint, RegNode, buildRegressionTree, predictReg, type Point } from './tree'

export type GBOptions = { numTrees: number; learningRate: number; maxDepth: number; minLeaf: number }

// ── Regression: additive trees fit to squared-loss residuals ─────────
export type GBRegressor = { init: number; lr: number; trees: RegNode[] }
export function trainGBRegressor(X: number[][], y: number[], opts: GBOptions): GBRegressor
//   F0 = mean(y); for m: r = y − F(X); tree = fit(X, r); F += lr·tree
export function gbPredict(model: GBRegressor, x: number[], nTrees?: number): number
//   init + lr · Σ_{b<nTrees} predictReg(trees[b], x)

// ── Classification: additive trees in log-odds space, logistic loss ──
export type GBClassifier = { init: number; lr: number; trees: RegNode[] } // init = log-odds(base rate)
export function trainGBClassifier(points: Point[], opts: GBOptions): GBClassifier
//   F0 = log(p̄/(1−p̄)); for m: p = σ(F); pseudo-residual = y − p; tree = fit(X, y−p); F += lr·tree
export function gbClassProb(model: GBClassifier, x: number[], nTrees?: number): number   // σ(F)
export function gbClassPredict(model: GBClassifier, x: number[], nTrees?: number): 0 | 1
```

Notes:
- The forest's `nTrees` pattern repeats: train the full ensemble once, and
  the widgets' round sliders pass `nTrees` to the (cheap) staged prediction —
  no retraining on drag. Boosting trees are sequential, so the first `m`
  trees *are* the ensemble after `m` rounds, exactly what the playback needs.
- Squared-loss residuals and logistic pseudo-residuals are the only two
  losses needed for the four widgets; the prose notes the framework extends
  to any differentiable loss.
- Deterministic given the data; weak learners are plain regression trees (no
  randomness needed for the core, though §8's stochastic boosting subsamples
  rows with a seeded rng).

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Data points | `var(--ink-muted)` faint | 1-D scatter (W1, W2) |
| True curve | `var(--ink-muted)` dashed | reference (W1, W2) |
| Ensemble prediction | `var(--accent)` | the boosted fit (W1, W2) |
| Residuals / next tree | `#c7522a` (orange) | W1 lower panel |
| Training error | `var(--ink-muted)` dashed | W3 (the misleading curve) |
| Held-out error | `var(--accent)` | W3; early-stop marker |
| Class 0 / 1 | `#3c5a8c` / `#c7522a` | W4 points + probability heatmap |
| Highlights / current | `var(--accent)` | new tree, current round, markers |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure numbering
14.1–14.4.

---

## Opening paragraph (draft, in voice)

> A good student does not reread the entire syllabus before each exam. They
> pull out the questions they got wrong last time and study *those* —
> correcting one specific weakness, then the next, then the next, until the
> mistakes run out. Gradient boosting is that habit turned into an algorithm.
> It builds its prediction not all at once but as a long sum of small
> corrections, and each correction is a little model trained on exactly the
> errors the sum has made so far.
>
> This is the mirror image of the last chapter. A random forest grows
> hundreds of trees in parallel, each blind to the others, and averages their
> votes to cancel out *variance*. Boosting grows its trees in strict
> sequence, each one trained on the residual mistakes of all the trees before
> it, and adds them up to cancel out *bias*. Same humble building block — a
> small decision tree — assembled by opposite philosophies, and the assembly
> matters as much as the block.
>
> By the end you will have watched a row of stumps converge on a curve no
> single stump could fit, seen that fitting residuals is quietly the gradient
> descent of Chapter 6 wearing a disguise, and learned why a boosted model,
> unlike a forest, must be told when to stop.

## Closing paragraph (draft, in voice)

> Gradient boosting closes the tree-based part of the book on its most
> powerful note. The same small tree that was a readable flowchart in Chapter
> 12, and a single unstable voice in a forest's crowd in Chapter 13, becomes
> here a single small correction in a long, deliberate sequence — and that
> sequence, tuned with a learning rate and stopped at the right moment, is
> the model that wins competitions on tabular data and quietly runs in
> production almost everywhere. Forests and boosting are the two great ways
> to turn many weak trees into one strong predictor: average them to tame
> variance, or sum them to tame bias.
>
> Part V leaves trees behind and returns to a single boundary — but not the
> casual straight line of Chapter 7. The next chapter asks a sharper
> question: of all the boundaries that separate the classes, which is the
> *best* one, the one that leaves the widest possible margin of safety on
> either side? The answer is one of the most elegant ideas in machine
> learning.
>
> ---
>
> *Next: Chapter 15 — Support vector machines.* Not just a separating line,
> but the one with the largest margin — and the kernel trick that lets a
> straight boundary curve.

---

## Expected scope

- Extend `lib/tree.ts`'s regression tree to p dimensions (and re-verify
  Chapter 12's RegressionTree widget) + one new lib (`lib/boosting.ts`, ~90
  lines) + four components.
- One MDX file (~800 lines).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 details, 5 "Show solution", sitemap entry).
- Bundle size: ~11–13 kB / ~272 kB First Load JS, in line with Ch 12/13.
- **Performance note:** train each ensemble once (memoised); round sliders
  pass `nTrees` to staged prediction. Verify a 100-round train on ~150 points
  stays well under a second before shipping.

Total estimated work: similar to Chapter 13 — four widgets, heavy reuse of
the Chapter-12 regression tree, one focused session.

---

## Open questions for sign-off

1. **Slug** — `14-gradient-boosting` (matches the landing title). Good?
2. **Four widgets as planned?** Residual-boosting (core) → learning-rate →
   overfitting/early-stop → 2-D boosting boundary. Three share the 1-D curve;
   the fourth is the 2-D classifier and the forest contrast. Happy with that
   split, or would you swap one (e.g. a dedicated *boosting-vs-forest*
   side-by-side instead of the boundary, or a *function-space gradient* view)?
3. **Regression-tree generalisation** — boosting's weak learners must fit
   residuals over 2-D features (Widget 4), so I'll generalise `lib/tree.ts`'s
   regression tree from 1-D to p-D (mirroring the Ch 13 classification-tree
   change) and re-verify Chapter 12's build. Flagging it as the one piece
   that touches shipped code; assume go unless you object.
```
