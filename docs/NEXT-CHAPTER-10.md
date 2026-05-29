# Design plan: Chapter 10 — Evaluation metrics

> Detailed plan for Chapter 10. Written before any code, per the recipe.
> The chapter the book has been promising since Chapter 7 §8 ("probability
> is information") and Chapter 9's closing line.

---

## Pedagogical role

Chapters 7, 8, and 9 all judged their classifiers by one number:
accuracy. Chapter 10 is where that habit gets dismantled. The reader
learns *why* accuracy is weak — sometimes actively misleading — and
picks up the real toolkit: the confusion matrix, precision/recall/F1,
the decision threshold as a free parameter, ROC/AUC, precision-recall
curves, and calibration.

This chapter is **evaluative, not algorithmic**. There's no new model
and no new optimiser. So the standard arc bends: instead of
concept → model → loss → training, it runs
problem → the object metrics derive from → the metric families → the
threshold that ties them together → the curves → the deepest property
(calibration). The running example is a logistic-regression-style
classifier — exactly the thing the reader built in Chapter 7 — so every
metric is computed on something already understood.

It pays off two long-standing threads:
- Chapter 7 §8 ("probability is information", calibration foreshadowed,
  "we will revisit ... in Chapter 10").
- Chapter 9's closing hook ("Next: Chapter 10 — Evaluation metrics").

And it sets up Chapter 11 (cross-validation): every metric here is
computed on one fixed dataset, which is its own kind of lie. Chapter 11
fixes *that*.

---

## The conceptual arc (11 sections)

1. **Why accuracy lies.** The rare-disease screen that scores 99% by
   answering "no" to everyone. Class imbalance makes accuracy a vanity
   metric. Hook + the chapter's promise.

2. **The confusion matrix.** Every classification metric is a ratio of
   four counts: true positives, false positives, false negatives, true
   negatives. Define them concretely on the running example. The matrix
   is the object; the metrics are summaries of it.

3. **Precision and recall.** Two questions: of the things I flagged, how
   many were right (precision)? Of the things I should have flagged, how
   many did I catch (recall)? Why they usually trade against each other.
   F1 as their harmonic mean, and why harmonic (punishes imbalance
   between the two).

4. **The threshold is yours to choose.** 0.5 is a default, not a law.
   Sliding the decision threshold walks precision and recall in opposite
   directions and changes every count in the matrix. → **Widget 1**
   (threshold strip + live confusion matrix + metrics, balanced vs
   imbalanced datasets). The centrepiece.

5. **ROC and AUC.** One threshold is one operating point; the ROC curve
   (TPR vs FPR) is the whole family of thresholds at once. AUC as
   threshold-independent *ranking quality* — the probability the model
   scores a random positive above a random negative. → **Widget 2** (ROC,
   marker rides the threshold, AUC shaded).

6. **Precision-recall curves.** Under heavy imbalance ROC looks
   optimistic — the huge true-negative pool flatters FPR. The PR curve
   tells the honest story because it never looks at true negatives. Same
   widget, toggled view; switch to the imbalanced dataset to see ROC stay
   rosy while PR collapses.

7. **Calibration.** A model that says 0.7 should be right 70% of the
   time. Reliability diagrams: bin by predicted probability, plot mean
   predicted vs observed frequency, compare to the diagonal. The deepest
   idea in the chapter: *ranking well and being calibrated are different
   things* — a model can have excellent AUC and untrustworthy
   probabilities. → **Widget 3** (reliability diagram, well-calibrated vs
   overconfident, with identical AUC shown on both). Pays off Ch 7 §8.

8. **Metrics for regression.** Brief breadth so the chapter isn't
   classification-only. MSE (penalises large errors quadratically), MAE
   (robust, same units as the target), R² (fraction of variance
   explained, and why it can go negative). ~250 words.

9. **Complexity.** Confusion counts and the per-threshold metrics are
   O(n). ROC, PR, and AUC need the scores sorted — O(n log n) — then one
   linear sweep. Calibration is one O(n) bucketing pass. Nothing here is
   expensive; the cost is always dominated by training the model in the
   first place.

10. **Implementing it yourself.** NumPy: confusion counts at a threshold,
    precision/recall/F1, the ROC sweep, trapezoidal AUC (plus the
    one-line rank-based equivalent), and calibration bins. ~30 lines.

11. **Problems.** Five, increasing difficulty (see below).

### Problems (sketch)

1. **Conceptual** — Why is a 99%-accurate model on a 1%-positive dataset
   potentially useless? What would precision and recall reveal that
   accuracy hides? (No code.)
2. **Implement the confusion matrix + precision/recall/F1** from scores
   and labels at a given threshold.
3. **Sweep the threshold** — compute precision and recall at 20
   thresholds, find the one maximising F1.
4. **Compute AUC two ways** — trapezoidal area under the ROC, and the
   rank-based Mann-Whitney form; verify they agree.
5. **Expose miscalibration** — take a well-ranked score set, distort it
   monotonically (preserving AUC), and show via calibration bins that AUC
   is unchanged while the reliability diagram bows away from the diagonal.
   The chapter's thesis as an exercise.

---

## The three visualisations (detailed)

All three are driven by a **single source of truth**: a `Scored[]` array
(a predicted score plus a true label per sample) produced by the
generators in `lib/metrics.ts`. Every count, curve, and bin is derived
from that array through the lib's pure functions, so nothing can desync.

### Widget 1 — `ThresholdConfusion` (build first; simplest)

- A horizontal axis from 0 to 1 of predicted probability. Two stacked
  1D histograms (or a jittered strip): negatives (blue `#3c5a8c`) and
  positives (orange `#c7522a`), so the reader sees the overlap.
- A **draggable vertical threshold line** (drag pattern borrowed from
  `SigmoidExplorer`'s horizontal-drag handler + a slider as backup).
- To the right (or below): a live **2×2 confusion matrix**, cells tinted
  — diagonal/correct in teal `var(--accent)`, off-diagonal/errors in
  warm `#c7522a` — with counts.
- A readout row: accuracy, precision, recall, F1.
- A **dataset toggle**: *balanced* (base rate 0.5) and *imbalanced* (base
  rate **0.05** — the always-negative model scores 95%, which reads as
  "great" while precision/recall are catastrophic; the dissonance is the
  lesson). Toggling the dataset **holds the current threshold fixed** (no
  reset to 0.5) so the reader can watch accuracy stay high while
  precision/recall crater at the *same* operating point.
- Pedagogical payoff: drag the threshold, watch precision and recall move
  in opposite directions; switch to imbalanced and watch accuracy lie.
- Threshold semantics identical to Widget 2: predict positive when
  score ≥ threshold; slider left = 0, right = 1; **default 0.5**.

### Widget 2 — `RocCurve` (ROC + PR toggle)

- Square plot. **ROC**: TPR (y) vs FPR (x), curve in `var(--accent)`, AUC
  as a shaded area beneath, chance diagonal dashed in `var(--rule)`.
- A **threshold slider** whose marker rides the curve — the explicit
  "one threshold = one point" mapping, reinforcing Widget 1's mental
  model. Same generators, same threshold semantics, same lib functions.
- A **view toggle ROC ↔ PR**. PR plots precision (y) vs recall (x), with
  the base-rate horizontal line as the no-skill baseline.
- A **dataset toggle** (balanced/imbalanced) shared in spirit with Widget
  1: on imbalanced, the ROC stays high (optimistic) while the PR curve
  visibly collapses (honest). The §6 "aha".
- Readout: AUC, and average precision for the PR view.

### Widget 3 — `CalibrationDiagram` (subtlest)

- Reliability diagram: x = mean predicted probability per bin, y =
  observed positive frequency per bin, diagonal dashed = perfect
  calibration. Bin dots sized by bin count; a faint histogram of bin
  counts underneath in muted ink.
- A **toggle: well-calibrated ↔ overconfident.** Both are the *same
  ranking* — built by drawing labels from a latent "true probability"
  and then either reporting that probability (calibrated) or pushing it
  toward the extremes through a monotonic distortion (overconfident).
- Because the distortion is monotonic, **AUC is identical for both** —
  displayed on the widget so the dissociation is explicit. The
  calibrated curve hugs the diagonal; the overconfident curve makes the
  classic S — below the diagonal on the right (claims 0.95, delivers
  0.8), above it on the left.
- **Show AUC and ECE side by side, on both toggles.** ECE (expected
  calibration error) is the count-weighted average gap between predicted
  and observed frequency across the bins — a few lines on top of
  `calibrationBins`. Toggling well-calibrated → overconfident should read
  as e.g. AUC 0.91 / ECE 0.02 → AUC 0.91 / ECE 0.14: *same ranking
  quality, very different trustworthiness*. The number nails the thesis
  shut where the picture only suggests it.
- This is the chapter's thesis in one picture: good ranking ≠ trustworthy
  probabilities.

---

## Files to create

```
lib/
  metrics.ts                    ← Scored type, confusion, precision/recall/
                                  F1/accuracy, ROC + AUC, PR + AP,
                                  calibration bins, score generators
components/
  ThresholdConfusion.tsx        ← Widget 1
  RocCurve.tsx                  ← Widget 2 (ROC + PR toggle)
  CalibrationDiagram.tsx        ← Widget 3
app/chapters/
  10-evaluation-metrics/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/10-evaluation-metrics'`) and
`app/sitemap.ts`. **Proposed slug: `10-evaluation-metrics`** (matches the
landing-page title "Evaluation metrics").

---

## Library design sketch — `lib/metrics.ts`

```ts
import { createRng, gauss } from './rng'
import { sigmoid } from './logistic'

/** A single scored example: model score in [0,1] plus the true label. */
export type Scored = { score: number; label: 0 | 1 }

export type Confusion = { tp: number; fp: number; fn: number; tn: number }

export type RocPoint = { fpr: number; tpr: number; threshold: number }
export type PrPoint = { recall: number; precision: number; threshold: number }

export type CalibrationBin = {
  lo: number
  hi: number
  count: number
  meanPredicted: number  // mean score of samples in the bin
  observedFreq: number   // fraction of positives in the bin
}

// ── Confusion + scalar metrics ─────────────────────────────────────
// Convention: predict positive when score >= threshold.
export function confusionAt(data: Scored[], threshold: number): Confusion
export function precision(c: Confusion): number   // tp / (tp + fp), 0 if undefined
export function recall(c: Confusion): number      // tp / (tp + fn)  (= TPR)
export function f1(c: Confusion): number          // harmonic mean of P and R
export function accuracy(c: Confusion): number    // (tp + tn) / total
export function fpr(c: Confusion): number         // fp / (fp + tn)

// ── Curves ─────────────────────────────────────────────────────────
// Sort by score desc once, sweep thresholds, group ties. Endpoints
// included so the curve spans (0,0)→(1,1) for ROC.
export function rocCurve(data: Scored[]): RocPoint[]
export function prCurve(data: Scored[]): PrPoint[]

// Trapezoidal area under the ROC, computed from the SAME points the
// widget draws (so the number and the picture never disagree).
export function aucFromRoc(roc: RocPoint[]): number

// Rank-based (Mann–Whitney) AUC — for the §10/problem cross-check.
export function aucRank(data: Scored[]): number

// Average precision (area under the PR curve).
export function averagePrecision(pr: PrPoint[]): number

// ── Calibration ─────────────────────────────────────────────────────
export function calibrationBins(data: Scored[], nBins: number): CalibrationBin[]

// Expected calibration error: count-weighted mean |meanPredicted −
// observedFreq| over non-empty bins. 0 = perfectly calibrated.
export function expectedCalibrationError(bins: CalibrationBin[]): number

// ── Generators (single source of truth for all three widgets) ───────
//
// Draw a latent "true probability" p = sigmoid(bias + N(0, spread)); draw
// the label ~ Bernoulli(p). By construction the score p is CALIBRATED:
// E[y | score = s] = s. `bias` sets the base rate (imbalance); `spread`
// sets how decisively the model separates the classes (i.e. the AUC).
//
// `distortion` reports a monotonic transform of p as the score while the
// label still comes from the true p — preserving the ranking (AUC) but
// breaking calibration. 'overconfident' pushes scores toward 0/1.
export type Distortion = 'none' | 'overconfident' | 'underconfident'

export function makeScores(
  n: number,
  seed: number,
  opts?: { baseRate?: number; spread?: number; distortion?: Distortion },
): Scored[]
```

Notes:
- `aucFromRoc` is the displayed AUC — it is trapezoidal over exactly the
  rendered points, so curve and number always agree (watch-out #2).
- `aucRank` exists for the §10 cross-check and Problem 4; it equals
  `aucFromRoc` up to floating-point and tie handling.
- The calibrated-vs-overconfident dissociation (watch-out #3) is built
  into `makeScores` via `distortion` — monotonic, so AUC is invariant.
- Reuses `createRng`/`gauss` from `lib/rng.ts` and `sigmoid` from
  `lib/logistic.ts`. No new dependencies.

---

## Visualisation colour palette

Reuse the book's established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Negative class (label 0) | `#3c5a8c` (blue) | negative histogram, FP/TN context |
| Positive class (label 1) | `#c7522a` (orange) | positive histogram, errors highlight |
| Correct cells | `var(--accent)` (teal) | confusion diagonal (TP, TN) tint |
| Error cells | `#c7522a` (orange) | confusion off-diagonal (FP, FN) tint |
| Curves & markers | `var(--accent)` | ROC/PR curve, AUC shaded fill (low alpha), current-threshold dot |
| Reference lines | `var(--rule)` dashed | ROC chance diagonal, PR base-rate line, calibration perfect diagonal |
| Axes / labels | `var(--ink-muted)` | ticks, captions |

Fonts per the design system: `font-sans` for UI/labels/captions,
`font-mono` for numeric readouts. Figure numbering 10.1, 10.2, 10.3.

---

## Opening paragraph (draft, in voice)

> A model that screens for a rare cancer can be right 99% of the time and
> still be worthless. If one person in a hundred carries the disease, a
> model that simply answers "no" to everyone — never flagging a single
> case — scores 99% accuracy. It has also never once done its job.
> Accuracy, the number we have leaned on for three chapters, has been
> quietly lying about how good our classifiers are.
>
> This chapter is about measuring classifiers honestly. Accuracy is one
> number, and the world is more complicated than one number. We will pull
> apart the kinds of mistake a classifier can make, see why precision and
> recall usually pull in opposite directions, watch a single tunable
> threshold trace out an entire family of models, and finish with the most
> under-appreciated property of all: whether a model that says "0.7" is
> right 70% of the time.
>
> By the end you will be able to look at a classifier and say not just how
> often it is right, but what it is right *about* — and whether you can
> trust the probabilities it hands you.

## Closing paragraph (draft, in voice)

> A classifier is not one number. It is a confusion matrix you can slide a
> threshold across, a curve that holds every threshold at once, and a
> promise about what its probabilities mean. Accuracy collapses all of
> that into a single digit and throws the rest away. The metrics in this
> chapter are what is left when you refuse to.
>
> But every metric here was computed on a single fixed dataset. That is
> its own kind of lie: a model can score beautifully on the data you
> happen to hold and fall apart on the data you don't. The next chapter is
> about measuring across data the model has never seen — and about
> choosing hyperparameters, like Chapter 9's λ, without fooling yourself.
>
> ---
>
> *Next: Chapter 11 — Cross-validation and tuning.* How to estimate a
> model's true performance honestly, and pick its knobs the right way.

---

## Expected scope

- One new lib module (`lib/metrics.ts`, ~180 lines) + three components.
- One MDX file (~750 lines).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX > 50, 5 problem
  headings, 5 details, 5 "Show solution", sitemap entry).
- Bundle size: ~9–10 kB / ~267 kB First Load JS, in line with Ch 7–9.

Total estimated work: one focused session, same as Chapters 7–9.
```
