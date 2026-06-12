# Design plan: Chapter 16 — Naive Bayes

> Detailed plan for Chapter 16, written before any code (per the recipe). The
> middle chapter of **Part V — More classical models**. After the SVM's pure
> geometry, a hard turn to pure probability: a classifier built almost
> entirely from Bayes' rule and one bold, plainly-false independence
> assumption — which, despite being wrong, is one of the fastest and most
> reliable baselines in machine learning, and the canonical spam filter.

---

## Pedagogical role

Every classifier so far has been **discriminative**: it draws a boundary
between the classes and models $P(y\mid x)$ directly — logistic regression,
the SVM, trees. Naive Bayes is the book's first **generative** classifier. It
turns the question around and asks, for each class, *what does the data from
this class look like?* — modelling $P(x\mid y)$ — then uses Bayes' rule to
flip that into a prediction. That reversal is the conceptual payload of the
chapter, and the cleanest possible illustration of the generative /
discriminative split that organises modern ML.

The "naive" part is a single, deliberately crude assumption: that within a
class, the features are *independent*. It is almost always false — words
co-occur, pixels correlate — and yet the resulting classifier is fast,
needs little data, and is startlingly hard to beat as a baseline, especially
on text. The chapter earns that paradox: it shows exactly what the assumption
throws away (the geometry widget), and exactly why the throwing-away rarely
hurts the final decision.

It compounds on the book:
- Chapter 3 (probability): Bayes' rule, priors, likelihoods, conditional
  independence — this is where that groundwork is spent.
- Chapter 7 (logistic regression): the discriminative twin. Naive Bayes and
  logistic regression are a famous generative/discriminative *pair* — under
  Gaussian NB's own assumptions the posterior is exactly the logistic /
  softmax form of Chapters 7–8, so the two models reach for the same boundary
  from opposite directions.
- Chapter 8 (softmax): the multi-class posterior comes free — Bayes' rule over
  $k$ classes is naturally multi-class, no one-vs-rest needed.
- Chapter 10 (calibration): NB's probabilities are notoriously *over-confident*
  precisely because the independence assumption double-counts correlated
  evidence — a concrete callback.

It sets up Chapter 17 (feature engineering): NB on text is the book's first
real encounter with the bag-of-words representation, and its failure modes
motivate thinking hard about features.

---

## The conceptual arc (12 sections)

1. **The reverend's classifier.** Sensory hook: a spam filter, or a medical
   test that is "99% accurate" yet usually wrong when it fires on a rare
   disease — the base-rate puzzle everyone gets backwards. Both are Bayes'
   rule. Naive Bayes is that one piece of arithmetic, scaled up into a
   classifier. Hook + promise (Bayes' rule, the naive assumption, the spam
   filter).

2. **Bayes' rule for classification.** Flip the question. Instead of modelling
   $P(y\mid x)$ directly, model how each class *generates* data, $P(x\mid y)$,
   and invert with Bayes:
   $$P(y\mid x) = \frac{P(x\mid y)\,P(y)}{P(x)} \propto P(x\mid y)\,P(y).$$
   Predict the class with the largest *posterior* — likelihood × prior. The
   denominator $P(x)$ is the same for every class, so it drops out of the
   argmax. This is the *generative* recipe, and the whole model.

3. **The trouble with the likelihood.** Why we cannot just do this. $P(x\mid
   y)$ is a joint distribution over all $p$ features at once; estimating it
   honestly needs a count for every *combination* of feature values — which
   explodes exponentially and is hopeless past a few features (the curse of
   dimensionality again). Bayes' rule is exact but the likelihood is
   intractable. We need a shortcut.

4. **The naive assumption.** The shortcut, and the chapter's namesake. *Assume
   the features are conditionally independent given the class:*
   $$P(x\mid y) = \prod_{j=1}^{p} P(x_j\mid y).$$
   Now each feature needs only its own little one-dimensional distribution per
   class — $p \times k$ easy estimates instead of one impossible joint. The
   assumption is almost always false, but it makes the model trivial to fit;
   the rest of the chapter is about cashing that in, and about how much the
   lie costs.

5. **Gaussian naive Bayes.** The continuous case. Model each feature, within
   each class, as a one-dimensional Gaussian — estimate a mean and variance
   per feature per class, multiply them, weight by the prior. The posterior
   becomes a sum of log-Gaussians, and the decision boundary is a clean conic
   curve. → **Widget 1: ClassConditional** (2-D blobs; the per-class
   axis-aligned Gaussian bells, the posterior heatmap, and the boundary, with
   the 1-D per-feature densities shown on the margins).

6. **What "naive" costs you.** The honest reckoning. Conditional independence
   forces each class's Gaussian to be *axis-aligned* — its contours are
   ellipses with no tilt, because a tilt *is* feature correlation. When the
   real data is correlated, NB fits the wrong-shaped bell and its boundary is
   visibly off. → **Widget 2: NaiveAssumption** (a correlation slider tilts the
   true data cloud; NB's fitted ellipse stubbornly stays axis-aligned, the
   boundary degrading as correlation grows — the lie made visible). The setup
   for §10's surprise: the decision often survives anyway.

7. **Multinomial naive Bayes and text.** The famous application. Represent a
   document as word counts (bag of words), model each class as a "bag" with a
   probability for each vocabulary word, and a document's likelihood is the
   product of its words' probabilities. This is the classic spam filter, and
   it is where naive Bayes genuinely shines. → **Widget 3: SpamFilter** (toggle
   words into a message and watch the running log-odds tip toward spam or ham,
   word by word — each word a piece of evidence added to the scale).

8. **Smoothing and log-space.** Two fixes that make it actually work. *The
   zero-frequency problem*: a single word never seen in the spam training set
   gives $P(\text{word}\mid\text{spam}) = 0$, which zeroes the entire product —
   one unseen word vetoes everything. *Laplace (add-one) smoothing* pads every
   count so nothing is ever exactly zero. *Underflow*: multiplying thousands of
   small probabilities rounds to zero in floating point, so we add
   log-probabilities instead of multiplying — turning the product into the
   running sum the widget already showed.

9. **Generative vs discriminative.** The Chapter-7 tie, made sharp. Under
   Gaussian NB's own assumptions, the posterior works out to *exactly* the
   sigmoid/softmax of logistic regression — the two models target the same
   boundary. The difference is how they get there: NB estimates the classes'
   distributions and inverts (generative); logistic regression optimises the
   boundary directly (discriminative). The classic consequence (Ng & Jordan):
   NB has higher asymptotic error but *converges faster* — it needs less data
   to reach its (lower) ceiling, which is why it is such a good baseline when
   data is scarce. → optional **Widget 4 / static figure: GenVsDisc** (the two
   boundaries on the same data, and a learning curve: NB wins at small $n$,
   logistic regression overtakes as $n$ grows).

10. **Why a false assumption still works.** The paradox resolved.
    Classification only needs the *argmax* of the posterior to be right, not
    the posterior itself. The independence assumption badly distorts the
    probability values — NB is famously over-confident, spitting out 0.999
    when it should say 0.7 (the Chapter-10 calibration callback) — but the
    *ranking* of classes is far more robust, so the decision is often correct
    even when the probability is nonsense. Plus a note on **complexity**:
    training is a single pass to count/average ($O(np)$), prediction is
    $O(pk)$ — about as fast as machine learning gets, and trivially online.

11. **Implementing it yourself.** Gaussian naive Bayes in ~25 lines (fit
    per-class means and variances; predict by summing log-Gaussians and the
    log-prior), then a multinomial-NB spam classifier on a tiny corpus to show
    the text path and Laplace smoothing.

12. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual — base rates and the generative idea.** (No code.) The "99%
   accurate" test for a 1-in-10 000 disease: compute the actual $P(\text{sick}
   \mid \text{positive})$ with Bayes' rule and explain why it is small. Then:
   what does it mean to call NB *generative*, and what exactly does the naive
   assumption assume — give a case where it is plainly false.
2. **Bayes' rule by hand.** A 2-feature, 2-class toy table of counts: compute
   the posterior for a query point under the naive assumption, with and
   without one feature, and see how each feature shifts the decision.
3. **Gaussian NB from scratch.** Fit per-class/per-feature means and
   variances on 2-D blobs; classify by summing log-Gaussians + log-prior;
   report accuracy and compare to the prior-only baseline.
4. **The spam filter (multinomial NB + smoothing).** Build a word-count NB on
   a small labelled corpus; classify new messages; show that an unseen word
   breaks it *without* Laplace smoothing and is fine *with* it. Connects §7–§8.
5. **Generative vs discriminative.** Train NB and logistic regression on the
   same data at growing training-set sizes; plot both learning curves and show
   the crossover — NB ahead when data is scarce, logistic regression ahead
   once data is plentiful. The Part-V capstone, tying back to Chapter 7.

---

## The visualisations (detailed)

Three core widgets plus an optional fourth (or static figure). New
`lib/naivebayes.ts`; reuse the class palette and the heatmap pattern.

### Widget 1 — `ClassConditional` (Gaussian NB; build first)

- A 2-D two- or three-class blob set. Each class is drawn as its fitted
  **axis-aligned Gaussian** (concentric elliptical contours), with the
  **posterior** rendered as the familiar blue↔orange heatmap and the **decision
  boundary** where the posterior crosses. On the top and right margins, the
  **1-D per-feature class densities** (the bells NB actually multiplies) so the
  reader sees the model is literally one Gaussian per feature per class.
- Optional: a slider on class-1's prior (the base rate) sliding the boundary,
  to make "× prior" tangible.
- Pedagogical job: this *is* Gaussian NB — independent per-feature bells,
  multiplied and weighted by the prior, give a smooth probabilistic boundary.

### Widget 2 — `NaiveAssumption` (what independence costs)

- One class of correlated 2-D data with a **correlation slider** that tilts the
  true cloud (rotates its covariance). Two ellipses overlaid: the **true**
  covariance (tilted, teal) and the **axis-aligned ellipse NB fits** (the only
  shape independence allows). As correlation grows, the two diverge and NB's
  decision boundary visibly drifts off the true one; a small accuracy readout
  quantifies the cost.
- Pedagogical job: "naive" is a picture — independence = no tilt. The model
  cannot represent correlation, full stop. Sets up §10's twist that the
  *decision* is often robust regardless.

### Widget 3 — `SpamFilter` (multinomial NB; the showpiece application)

- A small fixed vocabulary (~10–14 words) with hand-authored
  spam/ham likelihoods (e.g. *free, winner, meeting, money, project,
  click, …*). The reader **toggles words into a message**; the widget shows,
  word by word, each word's **log-likelihood-ratio** as a signed bar (toward
  spam or ham) and the **running total** sliding a needle from HAM to SPAM,
  with the final posterior probability. A **Laplace-smoothing toggle** shows an
  out-of-vocabulary or zero-count word breaking the classifier (−∞) without
  smoothing and behaving with it.
- Pedagogical job: naive Bayes as the thing it is most famous for — a spam
  filter — with the additive log-evidence story (§8) made literal and
  interactive. The most engaging widget in the chapter.

### Widget 4 — `GenVsDisc` (optional; generative vs discriminative)

- The same 2-D data classified by **Gaussian NB** and by **logistic
  regression**, their two boundaries overlaid (often close, sometimes
  tellingly different), beside a **learning curve**: test accuracy versus
  training-set size for both, NB above at small $n$, logistic regression
  overtaking as $n$ grows.
- Pedagogical job: the generative/discriminative pairing and the
  small-data-vs-large-data trade — the Chapter-7 callback and Part-V capstone.
- *Alternative:* demote to a static two-panel figure if four interactive
  widgets is one too many for this chapter.

---

## Files to create

```
lib/
  naivebayes.ts                 ← NEW: Gaussian NB (fit/predict, log-posterior),
                                  multinomial NB (counts, Laplace, log-odds),
                                  a correlated-blob generator, the toy text
                                  corpus / vocabulary.
components/
  ClassConditional.tsx          ← Widget 1
  NaiveAssumption.tsx           ← Widget 2
  SpamFilter.tsx                ← Widget 3
  GenVsDisc.tsx                 ← Widget 4 (or a static figure component)
app/chapters/
  16-naive-bayes/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/16-naive-bayes'`) and
`app/sitemap.ts`. **Proposed slug: `16-naive-bayes`** (matches the landing
title "Naive Bayes").

Reused: `makeBlobs2` and the `Point` type (`lib/tree.ts`), `trainLogistic` /
`makeBlobs` (`lib/logistic.ts`) for the gen-vs-disc widget, the class palette,
the heatmap + 1-D-curve patterns. No new dependencies.

---

## Library design sketch

```ts
// ── lib/naivebayes.ts — new ───────────────────────────────────────────
import type { Point } from './tree'

// ── Gaussian naive Bayes ─────────────────────────────────────────────
export type GaussianNB = {
  classes: number[]
  prior: number[]          // P(y=c)
  mean: number[][]         // mean[c][j]
  var: number[][]          // var[c][j]  (per-feature, the diagonal)
}
export function fitGaussianNB(data: Point[]): GaussianNB
export function gnbLogPosterior(m: GaussianNB, x: number[]): number[]  // per class
export function gnbPredict(m: GaussianNB, x: number[]): number

// ── Multinomial naive Bayes (text) ───────────────────────────────────
export type MultinomialNB = {
  vocab: string[]
  logPrior: number[]                 // per class
  logLik: number[][]                 // logLik[c][word], Laplace-smoothed
}
export function fitMultinomialNB(docs: number[][], y: number[], vocab: string[], alpha?: number): MultinomialNB
export function mnbLogOdds(m: MultinomialNB, counts: number[]): number   // log P(spam|·)/P(ham|·)

// ── Datasets ─────────────────────────────────────────────────────────
export function makeCorrelatedBlobs(n: number, seed: number, rho: number): Point[]
//   two classes, each a Gaussian with correlation rho (tilt) — for Widget 2.

// ── Toy text corpus for the spam widget / problems ───────────────────
export const SPAM_VOCAB: string[]
export const SPAM_LIKELIHOODS: { spam: number[]; ham: number[] }  // hand-authored, deterministic
```

Notes & numerical-verification plan (run in Node *before* building widgets):
- **Gaussian NB:** verify accuracy on `makeBlobs2` matches expectation
  (high on separable blobs), that the boundary is the right conic, and that
  per-class means/vars are sane. Confirm the multi-class posterior sums to 1
  after softmax-normalising the log-posteriors.
- **The naive cost:** with `makeCorrelatedBlobs`, confirm NB accuracy *drops*
  as $\rho \to 1$ while a full-covariance model would hold — quantifying
  Widget 2.
- **Multinomial NB:** on the toy corpus, verify spam/ham messages classify
  correctly, that a zero-count word sends the unsmoothed log-odds to $\pm\infty$
  (or NaN) and Laplace smoothing fixes it, and that the per-word log-odds sum
  to the total (the additive-evidence invariant Widget 3 draws).
- **Gen vs disc:** confirm the learning-curve crossover (NB ahead at small
  $n$, logistic regression ahead at large $n$) actually appears on the chosen
  dataset/seed before committing to Widget 4; if it is flaky, demote to a
  static figure.
- **Performance:** everything here is closed-form one-pass — sub-millisecond.
  Heatmaps are the only cost and follow the established coarse-grid pattern.
- **Determinism:** seeded RNG for datasets, hand-authored (not sampled) text
  likelihoods — no `Math.random`, no `any`, British spelling throughout.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Class 0 / 1 (/ 2) | `#3c5a8c` / `#c7522a` (/ `#5d8a3a`) | points, densities, heatmap |
| Posterior heatmap | blue↔orange alpha by confidence | W1, W4 |
| Decision boundary | `var(--accent)` | W1, W4 |
| True covariance ellipse | `var(--accent)` | W2 |
| NB axis-aligned ellipse | `var(--ink-muted)` dashed | W2 |
| Spam evidence / Ham evidence | `#c7522a` / `#3c5a8c` | W3 signed bars + needle |
| Per-feature 1-D densities | class colours, faint fill | W1 margins |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure numbering
16.1–16.4 (the gen-vs-disc figure numbered whether interactive or static).

---

## Opening paragraph (draft, in voice)

> A test for a rare disease is, the leaflet promises, "99% accurate." You test
> positive. Most people read that number and panic — and most people are
> wrong, often wildly, because the leaflet's 99% answers a different question
> from the one that matters. The question you care about is *given that I
> tested positive, what is the chance I am actually sick?* — and if the
> disease is rare enough, the answer can be a reassuring few percent even after
> a positive test. The gap between those two numbers is the single most
> misunderstood idea in probability, and closing it takes exactly one formula:
> Bayes' rule.
>
> Naive Bayes is that formula scaled up into a classifier. Where logistic
> regression and the support vector machine drew a boundary and asked which
> side a point fell on, naive Bayes does something stranger and older. For each
> class it builds a little model of *what that class's data looks like* — what
> a spam email tends to contain, what a healthy blood sample tends to read —
> and then, faced with a new point, it asks which class was most likely to
> have produced it. It reasons backwards from effect to cause, which is what
> Bayes' rule is for.
>
> The catch is in the name. To make the backwards reasoning tractable, naive
> Bayes assumes every feature is independent of every other within a class — an
> assumption so brazenly false that it is almost funny, and yet the classifier
> it produces is fast, frugal with data, and very hard to beat as a first
> attempt, especially on text. By the end of this chapter you will have built a
> working spam filter out of nothing but counting and Bayes' rule, seen exactly
> what the naive assumption throws away, and understood the lovely paradox at
> the centre of the method: that being wrong about the probabilities can still
> mean being right about the answer.

## Closing paragraph (draft, in voice)

> Naive Bayes is the bargain of machine learning: one pass over the data to
> count and average, an assumption everyone knows is false, and in return a
> classifier that trains in a blink, runs on a phone, needs barely any data,
> and embarrasses far fancier models on text. It earns its keep not by being
> right about the world — its independence assumption is a fiction and its
> confident probabilities are not to be trusted — but by being right about the
> *decision* often enough to matter. It is the baseline you should beat before
> you believe anything more complicated is helping.
>
> It also closes a loop. Naive Bayes and the logistic regression of Chapter 7
> are the same boundary approached from opposite sides — model the classes and
> invert, or model the split directly — the generative and discriminative faces
> of the same coin, and seeing them together is the real lesson of Part V.
>
> ---
>
> *Next: Chapter 17 — Feature engineering.* Every model in the book so far has
> taken its features as given. The next chapter asks where they come from — the
> unglamorous, decisive craft of turning raw data into the columns a model can
> actually use, where more problems are won and lost than in any choice of
> algorithm.

---

## Expected scope

- One new lib (`lib/naivebayes.ts`, ~140 lines: Gaussian NB, multinomial NB,
  correlated-blob generator, toy corpus) + three or four components.
- One MDX file (~850 lines, 12 sections, 5 problems).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 `<details>`, 5 "Show solution", sitemap entry, no American
  spellings, no `Math.random`, no `any`).
- Bundle size: ~10–13 kB / ~270 kB First Load JS, in line with Ch 14/15.
- **Numerical verification first:** a Node port confirming Gaussian-NB
  accuracy and boundary, the correlation cost, the multinomial spam
  classification + Laplace fix + additive-evidence invariant, and the
  gen-vs-disc crossover — before any component is built.

Total estimated work: a touch lighter than Chapter 15 (closed-form, no
iterative solver), with the text/spam widget as the one genuinely new
component pattern (signed evidence bars + a needle, rather than a 2-D scene).

---

## Open questions for sign-off

1. **Slug** — `16-naive-bayes` (matches the landing title). Good?
2. **Widgets** — three core (ClassConditional → NaiveAssumption → SpamFilter)
   plus a fourth, GenVsDisc (NB vs logistic regression + learning-curve
   crossover). Keep GenVsDisc as a **fourth interactive widget**, or make it a
   **static two-panel figure** and stop at three interactive ones? (I lean
   interactive — the crossover is worth dragging — but it is the most likely
   cut.)
3. **The spam widget's corpus** — I'll hand-author a small (~12-word)
   vocabulary with fixed, sensible spam/ham likelihoods so the demo is
   deterministic and legible, rather than sampling a synthetic corpus.
   Assume go unless you object.
```
