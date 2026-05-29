# Design plan: Chapter 11 — Cross-validation and tuning

> Detailed plan for Chapter 11, written before any code (per the recipe).
> Closes Part III (Evaluating models). Directly foreshadowed by Chapter
> 10's closing line and pays off Chapter 9 §8 ("choosing λ") and Chapter
> 10 Problem 5 (holdout validation).

---

## Pedagogical role

Chapter 10 taught the reader *which* number to measure. Chapter 11 teaches
them to measure it *honestly* — on data the model has never seen — and to
use that honest estimate to choose a model's settings. It is the discipline
chapter: hold data back, never tune on what you will test on, trust the
average over many splits.

It is not an algorithm chapter in the way Chapters 6–9 were. There is no new
model. The new machinery is a *protocol* — the train/test split, k-fold
cross-validation, and the train/validation/test discipline — plus the small
amount of code that implements it. Because of that it reuses the book's
existing libraries almost entirely: the polynomial-ridge tools from
`lib/regularisation.ts`, the logistic-regression trainer from the same file,
and the metrics from `lib/metrics.ts`. The only genuinely new lib is the
split/cross-validate harness.

It pays off two threads explicitly:
- Chapter 9 §8 ("Choosing λ ... is cross-validation ... Chapter 11 covers
  this in detail") and Chapter 9's overfitting story.
- Chapter 10's closing beat ("every metric here was computed on a single
  fixed dataset ... its own kind of lie").

And it sets up the rest of the book: every model from Part IV onward (trees,
forests, SVMs, networks) is compared and tuned with this chapter's machinery.

---

## The conceptual arc (11 sections)

1. **The optimism of training error.** A model graded on its own training
   data flatters itself. Chapter 9's degree-15 polynomial had near-zero
   training error and was garbage between the points. Training performance
   ≠ generalisation performance, and conflating them is the commonest way to
   fool yourself. Hook + promise.

2. **The train/test split.** Hold back a random slice the model never sees
   during fitting; score there. The held-out score is an (unbiased) estimate
   of generalisation. State the protocol and the one inviolable rule: the
   model never learns from the test data.

3. **The optimism gap, made visible.** Sweep model complexity (polynomial
   degree). Training error falls monotonically toward zero; held-out error
   falls, bottoms out, then climbs — the U-shape. The gap between the two
   curves *is* overfitting, measured. → **Widget 1** (train vs test error vs
   complexity).

4. **One split is not enough.** A single split is *unbiased but
   high-variance*: the estimate jumps around depending on which points
   happened to land in the test set, and the held-out data never gets to
   improve the model. Show the estimate wobbling as the split seed changes.

5. **k-fold cross-validation.** Partition into k folds; each fold is the test
   set exactly once while the other k−1 train. Average the k scores. Every
   point is tested once and trains k−1 times — lower variance, no wasted
   data. → **Widget 2** (k-fold partition: the held-out band moves through
   the folds, per-fold scores accumulate, the mean ± spread settles).

6. **Choosing hyperparameters with CV.** The real payoff: pick λ (Chapter 9),
   or polynomial degree, or k in kNN, by the cross-validated score. The
   CV-score-vs-hyperparameter curve has a sweet spot; the training score does
   not (it just keeps improving). → **Widget 3** (CV score vs λ with
   per-fold error bars, train score shown rising for contrast, the pick
   marked).

7. **The leakage trap: never tune on the test set.** If you choose the
   hyperparameter by the test score, the test score is no longer honest — you
   have tuned *to* it, and it is now optimistic. You need a separate
   validation set (train/validation/test), or *nested* CV. The discipline:
   the test set is touched exactly once, at the very end.

8. **Practical concerns.** Stratified folds (preserve class balance — vital
   for the imbalanced problems of Chapter 10). The standard choice of k (5 or
   10) and the bias/variance/cost trade-off behind it. Leave-one-out as the
   k = n limit. The subtlest leak: any data-dependent preprocessing
   (standardisation, feature selection) must be fit *inside* each fold, on
   the training part only — fitting it on the whole dataset first leaks the
   test data and inflates the score.

9. **Complexity.** CV costs k× a single fit. Grid search over g
   hyperparameter values costs g·k fits; nested CV multiplies again by the
   outer fold count. Random search reaches good settings in far fewer fits
   than grid search when only a few hyperparameters matter. The honest
   estimate is never free — you pay for it in compute.

10. **Implementing it yourself.** NumPy: a k-fold index split, a
    `cross_val_score` loop, and a grid search that picks the best λ by mean
    CV accuracy — reusing the logistic regression from earlier chapters.
    ~30 lines.

11. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual** — Why is training error a biased (optimistic) estimate of
   generalisation error, while a single train/test split is unbiased but
   high-variance? What does k-fold buy over a single split? (No code.)
2. **Implement a k-fold split** — `k_fold_indices(n, k, seed)` returning
   train/test index arrays; verify the folds are disjoint and every index is
   tested exactly once.
3. **Implement `cross_val_score`** — given fit/predict on a logistic-
   regression task, return per-fold accuracy and the mean ± std.
4. **Tune λ by CV, and see the leak** — grid-search λ by k-fold CV and report
   the pick; then "tune on the test set" instead and show the test-tuned
   estimate is optimistically biased (looks better than the honest CV pick
   delivers).
5. **Feature-selection leakage** — with many pure-noise features, select the
   features most correlated with the label *on the whole dataset* before CV,
   and watch CV accuracy climb well above chance on noise; then do the
   selection *inside* each fold and watch it fall back to ~0.5. The chapter's
   deepest practical lesson, as code.

---

## The three visualisations (detailed)

Distinct one-new-idea-each progression: Widget 1 establishes *that* training
error lies (one split); Widget 2 shows *the k-fold mechanism* that gives a
stable estimate; Widget 3 *applies* k-fold CV to pick a hyperparameter.

### Widget 1 — `OptimismGap` (build first; simplest)

- Reuses the polynomial-ridge tooling from `lib/regularisation.ts`
  (`makeNoisyScatter`, `polyFeatures`, `ridgeRegression`, `evalPoly`).
- A noisy 1-D dataset is split once into train and test. X-axis = polynomial
  degree (1–15); two curves: **training error** (`--ink-muted`, the
  misleading one, downplayed) falling toward zero, and **test error**
  (`--accent`, the honest one, highlighted) tracing a U.
- A degree slider with a marker on both curves; the sweet-spot degree
  (min test error) called out. A small inset or second panel optionally shows
  the actual fitted curve at the chosen degree (under/good/over-fit), but the
  two-error-curves plot is the core.
- Payoff: training error always improves with complexity; only the held-out
  error has a minimum, and that minimum is the model you want.

### Widget 2 — `KFoldCV` (the mechanism)

- A classification dataset (`makeSparseDataset` from `lib/regularisation.ts`,
  a few features) fit by `trainLogisticL2`, scored by accuracy
  (`lib/metrics.ts`).
- A horizontal bar of the n example-indices, partitioned into **k folds**
  (k slider, 2–10). Step/play through the folds: the current fold is the
  held-out **test band** (`--accent` tint), the rest are **training**
  (`--rule`/muted). As each fold resolves, its score appears; the running
  **mean ± std** of the per-fold scores settles as all k complete.
- A toggle or side-by-side: the single-split estimate (one fold's score)
  visibly wobbles vs the stable k-fold mean — the §4 → §5 payoff.
- Payoff: every point is tested exactly once; the average over folds is a
  lower-variance estimate than any single split.

### Widget 3 — `CVTuning` (the application; synthesis)

- Sweeps a hyperparameter — **λ** for L2 logistic regression
  (`trainLogisticL2`), tying directly back to Chapter 9 — over a log grid.
- For each λ, runs k-fold CV and plots the **mean CV score** with **per-fold
  error bars**. The **training score** is overlaid (rising monotonically as
  λ → 0) to contrast with the CV score, which peaks then falls. A vertical
  marker shows the **chosen λ** (best mean CV score).
- A k slider (folds) and a "show training score" toggle.
- Payoff: the training score would tell you to set λ = 0 (most flexible); the
  CV score tells you the truth — there is a sweet spot, and CV finds it
  without ever touching a test set.

---

## Files to create

```
lib/
  crossval.ts                   ← trainTestSplit, kFoldSplit, crossValidate
                                  (generic harness; pure + testable)
components/
  OptimismGap.tsx               ← Widget 1
  KFoldCV.tsx                   ← Widget 2
  CVTuning.tsx                  ← Widget 3
app/chapters/
  11-cross-validation-and-tuning/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/11-cross-validation-and-tuning'`)
and `app/sitemap.ts`. **Proposed slug: `11-cross-validation-and-tuning`**
(matches the landing-page title "Cross-validation and tuning").

No new dependencies. Heavy reuse of `lib/regularisation.ts`,
`lib/metrics.ts`, `lib/rng.ts`.

---

## Library design sketch — `lib/crossval.ts`

```ts
import { createRng } from './rng'

export type Fold = { trainIdx: number[]; testIdx: number[] }

/** Shuffle 0..n-1 with a seeded Fisher–Yates (uses createRng). */
function shuffledIndices(n: number, seed: number): number[]

/** Single random split: testFrac of the indices held out. */
export function trainTestSplit(
  n: number,
  testFrac: number,
  seed: number,
): Fold

/**
 * k-fold partition. Shuffle once, cut into k contiguous folds; fold i is the
 * test set, the rest train. Every index appears in exactly one test fold.
 */
export function kFoldSplit(n: number, k: number, seed: number): Fold[]

/**
 * Generic cross-validation harness. `fit` builds a model from the training
 * indices; `score` evaluates it on the test indices. Returns per-fold scores
 * and their mean and standard deviation. Model-specific glue (which trainer,
 * which metric) lives in the component — only the protocol lives here.
 */
export function crossValidate<M>(
  folds: Fold[],
  fit: (trainIdx: number[]) => M,
  score: (model: M, testIdx: number[]) => number,
): { foldScores: number[]; mean: number; std: number }
```

Notes:
- `kFoldSplit` guarantees disjoint test folds covering every index exactly
  once — the property Problem 2 asks the reader to verify.
- `crossValidate` is generic over the model type so the same harness drives
  the polynomial widget (ridge model) and the logistic widgets (L2 model).
- Everything is deterministic given the seed (no `Math.random`), consistent
  with the rest of the book.

Reused, not rewritten: `polyFeatures`, `ridgeRegression`, `evalPoly`,
`makeNoisyScatter`, `trueCurve`, `trainLogisticL2`, `makeSparseDataset`,
`LRPoint` from `lib/regularisation.ts`; `accuracy`/`confusionAt`/`rocCurve`/
`aucFromRoc` from `lib/metrics.ts`.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Training score/error | `var(--ink-muted)` | the *misleading* curve — deliberately downplayed |
| Held-out / CV score | `var(--accent)` (teal) | the *honest* curve — highlighted; error bars; sweet-spot marker |
| Held-out fold band | `var(--accent)` low-alpha | the current test fold in the k-fold bar |
| Training folds | `var(--rule)` / muted | the k−1 training folds |
| Reference lines | `var(--rule)` dashed | chance level, chosen-value vertical |
| Class scatter (if shown) | `#3c5a8c` / `#c7522a` | class 0 / class 1, per the book |
| Axes / labels | `var(--ink-muted)` | ticks, captions |

Fonts per the design system: `font-sans` UI/labels, `font-mono` numeric
readouts. Figure numbering 11.1, 11.2, 11.3.

---

## Opening paragraph (draft, in voice)

> A model graded on the very data it learned from is a student who saw the
> exam in advance. Of course it scores well — it has had every chance to
> memorise the answers. Chapter 9's degree-fifteen polynomial passed through
> all twenty-five of its training points and predicted nonsense in the gaps
> between them; its training error was nearly zero and its real error
> enormous. Training performance is not the performance you will get on data
> you have not seen, and confusing the two is the most common way to fool
> yourself in machine learning.
>
> This chapter is about not fooling yourself. We hold data back, measure on
> it honestly, and find that a single held-out slice is a noisy and wasteful
> way to do so. The fix — cross-validation — uses every data point as both
> training and test, never at the same time, and it is the engine behind
> almost all honest model comparison and hyperparameter tuning. It is also,
> at last, how you would actually choose Chapter 9's λ.
>
> By the end you will be able to estimate how well a model will really do,
> and choose its settings, without ever once peeking at the data you have set
> aside to judge it.

## Closing paragraph (draft, in voice)

> Cross-validation is less an algorithm than a discipline: hold data back,
> never tune on what you will test on, and trust the average over many splits
> more than any single number. Every model in the rest of this book — trees,
> forests, support vector machines, networks — is compared and tuned with the
> machinery of this chapter. It is the referee that keeps the comparisons
> fair.
>
> And so far every model has drawn a straight line — or a straight line in
> some transformed space. The next part of the book abandons that shape
> entirely. A decision tree carves the input space into rectangles by asking
> one yes-or-no question at a time, fitting relationships no linear model can
> express — at the price of a vivid new way to overfit, which the
> cross-validation you just learned is exactly the tool to catch.
>
> ---
>
> *Next: Chapter 12 — Decision trees.* A model that splits the world into
> boxes with a sequence of simple questions, and the greedy algorithm that
> grows it.

---

## Expected scope

- One new lib module (`lib/crossval.ts`, ~90 lines) + three components.
- One MDX file (~750 lines).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX > 50, 5 problem headings,
  5 details, 5 "Show solution", sitemap entry).
- Bundle size: ~8–9 kB / ~266 kB First Load JS, in line with Ch 9–10.

Total estimated work: one focused session, same as Chapters 8 and 10.
```
