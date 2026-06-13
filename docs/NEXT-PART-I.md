# Design plan: Part I — Foundations (Chapters 1–5)

> Combined plan for the five foundation chapters, written before any code (per
> the recipe). These open the book and were left for last deliberately: now that
> Parts II–VII exist, the foundations can point *forward* to concrete chapters
> ("we build this in Chapter 6", "you will meet this again in Chapter 17")
> instead of gesturing vaguely. The arc: what learning *is* (1), the workflow and
> the central danger of overfitting (2), the minimal maths (3), the first model
> you can hold in your head (4: kNN), and the first model you fit (5: linear
> regression) — which hands straight off to Chapter 6's gradient descent.

Each chapter keeps the house style: opening hook, numbered sections, runnable
Python, five problems with collapsible solutions, a "Next" hook. Foundation
chapters lean a little more on prose and run 2–3 widgets (vs 4–5 for the
algorithm chapters), per the HANDOFF guidance that Chapter 1 especially is
"mostly prose".

Heavy reuse, almost no new lib code:
- **kNN** (`knnPredict`, `knnNeighbours`, `fitScaler`) — `lib/features.ts`.
- **Least squares / polynomials** (`polyFeatures`, `ridgeRegression` with λ = 0,
  `solveLinear`, `evalPoly`, `makeNoisyScatter`, `trueCurve`) —
  `lib/regularisation.ts`.
- **Datasets** (`makeBlobs`/`makeMoons`, `makeBlobs2`/`makeMoons2`,
  `makeScaleData`) — `lib/logistic.ts`, `lib/tree.ts`, `lib/features.ts`.
- One tiny new helper file `lib/foundations.ts` only if needed (a 1-D regression
  generator and a least-squares-line fit; both can also be inlined).

---

## Chapter 1 — What is machine learning?  (`1-what-is-machine-learning`)

**Role.** The front door. Define machine learning by contrast with ordinary
programming (rules written by hand vs rules *learned from examples*), introduce
the organising triad the whole book uses — **model · loss · optimiser** — and the
supervised / unsupervised / (a word on reinforcement) divide. Establish
*generalisation* as the real goal. Mostly prose; two light widgets.

**Arc (≈10 sections).** A program that learns → rules vs examples (the spam
filter you do not write by hand) → the ingredients: a model (a function with
knobs), a loss (how wrong), an optimiser (turn the knobs) → supervised learning
(labels) → unsupervised learning (no labels) → a word on reinforcement →
generalisation is the point (not memorising the training data) → a map of the
book → it is all the same three ideas → problems.

**Widgets (2).**
1. `LearnFromExamples` — a scatter of points and a model fit through them; a
   model-flexibility slider (a line → a gentle curve → a wiggly curve) showing a
   model "learning the rule" from examples, and previewing that too much
   flexibility just memorises. The whole idea of fitting, in one picture.
2. `SupervisedVsUnsupervised` — the same points shown two ways: **labelled**
   (coloured by class, a boundary drawn — supervised) and **unlabelled** (all one
   colour, grouped by proximity — unsupervised). A toggle flips between the two
   framings of the *same* data, making the distinction concrete.

**Reuse.** `makeNoisyScatter`/`ridgeRegression` (W1); `makeBlobs2` + the existing
kNN or a simple grouping (W2). Closes → Chapter 2 (the workflow).

---

## Chapter 2 — The ML workflow  (`2-ml-workflow`)

**Role.** How a model is actually built and trusted: collect data, **split** into
train and test, fit on train, evaluate on held-out test, and confront the central
villain of the whole subject — **overfitting**. Introduces underfitting vs
overfitting, the bias–variance intuition, and why the test set is sacred. This is
the conceptual backbone every later chapter leans on.

**Arc (≈11 sections).** The goal is generalisation, not memorisation → the
train/test split → fitting on the training set → the test set is the future →
underfitting (too simple) → overfitting (too complex, memorises noise) → the
classic U: error vs model complexity → bias and variance, plainly → the
validation set and a word on cross-validation (preview of Chapter 11) → never
touch the test set twice → problems.

**Widgets (3).**
1. `TrainTestSplit` — a scatter split into train (filled) and test (hollow); a
   model fit on train only, with **train error** and **test error** read out
   separately, making the held-out idea tangible.
2. `OverfitDegree` — the canonical picture: noisy 1-D data and a polynomial fit
   with a **degree slider**; at low degree it underfits, at high degree it
   wiggles through every point and the **test error explodes** while train error
   keeps falling. The over/under-fitting U made interactive — the single most
   important figure in the book.
3. `ComplexityCurve` — train vs test error plotted against model complexity (the
   U-curve itself), the sweet spot marked, tied to the fits above.

**Reuse.** `makeNoisyScatter`, `polyFeatures`, `ridgeRegression`/`solveLinear`,
`evalPoly` — all already in `lib/regularisation.ts`. Closes → Chapter 3.

---

## Chapter 3 — A mathematical toolkit  (`3-mathematical-toolkit`)

**Role.** The minimal maths the book uses, taught geometrically and *only* what is
needed, with each idea tagged to where it pays off later. Three strands: a little
**linear algebra** (vectors, dot products, matrices as transformations), a little
**calculus** (derivative as slope, gradient as steepest-ascent direction — the
fuel of Chapter 6), and a little **probability** (distributions, expectation,
Bayes' rule — the fuel of Chapters 7 and 16).

**Arc (≈11 sections).** Why a little maths → vectors are arrows / data are points
→ the dot product measures alignment (and is the weighted sum in every linear
model) → matrices transform space → the derivative is a slope → the gradient
points uphill (and we will walk *down* it in Chapter 6) → randomness and
distributions → mean and variance → Bayes' rule, the engine of belief-updating →
how each tool returns later → problems.

**Widgets (3).**
1. `DotProduct` — two draggable vectors with their dot product shown live, plus
   the projection of one onto the other; positive when aligned, zero when
   perpendicular, negative when opposed. The geometry behind $w \cdot x$.
2. `SlopeAndGradient` — left: a 1-D curve with a draggable point and its tangent
   line (the derivative as slope); right: a 2-D surface (contours) with the
   gradient arrow at a draggable point pointing straight uphill. Foreshadows
   gradient descent.
3. `DistributionExplorer` — a Gaussian with draggable mean and adjustable
   standard deviation over a histogram of samples, with mean/variance read out;
   the shape every "noise" and "likelihood" in the book assumes.

**Reuse.** `computeContours` (`lib/contours.ts`) + a `LossFn` from
`lib/gradient.ts` for W2; `gauss` (`lib/rng.ts`) for W3. Closes → Chapter 4.

---

## Chapter 4 — Your first model: k-nearest neighbours  (`4-k-nearest-neighbours`)

**Role.** The first actual model, chosen because it needs no equations: to
classify a point, look at its $k$ nearest labelled neighbours and take a vote.
Builds the core intuitions — a **decision boundary**, the **bias–variance** effect
of $k$ (small $k$ overfits, large $k$ underfits — a concrete instance of Chapter
2), why **scaling matters** (Chapter 17 preview), and the **curse of
dimensionality** and laziness/cost that motivate everything cleverer to come.

**Arc (≈11 sections).** Prediction by analogy → the algorithm (find $k$ nearest,
vote) → distance, and why scale matters → the decision boundary it carves → the
role of $k$: from jagged to smooth → choosing $k$ (the validation idea, Chapter
11) → regression with kNN (average the neighbours) → it is a *lazy* learner (no
training, slow prediction) → the curse of dimensionality → where kNN is and is not
the right tool → from-scratch → problems.

**Widgets (3).**
1. `KNNBoundary` — a 2-D two-class scatter with the kNN decision-boundary heatmap
   and a **k slider**; small $k$ gives a jagged, island-pocked boundary (low
   bias, high variance), large $k$ a smooth one (high bias) — the bias–variance
   trade-off you can *drag*.
2. `KNNVote` — a movable query point showing its $k$ nearest neighbours
   highlighted and the vote tally, so "k nearest, take a vote" is literal.
3. `KNNScaling` — the same data with one feature on a large scale; raw distance is
   dominated by it (boundary ignores the other feature), and a **standardise**
   toggle fixes it — the Chapter-17 lesson, met early. (Reuses `makeScaleData` /
   `knnNeighbours` / `fitScaler` directly.)

**Reuse.** `knnPredict`, `knnNeighbours`, `fitScaler`, `makeScaleData`
(`lib/features.ts`); `makeBlobs2`/`makeMoons2` (`lib/tree.ts`). Closes → Chapter 5.

---

## Chapter 5 — Linear regression  (`5-linear-regression`)

**Role.** The first model you *fit by minimising a loss* — the mother of all
models, and the direct setup for Chapter 6. The line of best fit, **residuals**,
the **mean-squared-error** loss, the **normal-equation** closed form, and what the
fit means ($R^2$, interpreting coefficients). Ends pointing at the question
Chapter 6 answers: what if there is no closed form?

**Arc (≈11 sections).** Fitting a line to a trend → the model $\hat y = w x + b$ →
residuals, the misses → the loss: mean squared error → minimising it: the line
that balances the residuals → the closed form (the normal equations) → multiple
features (a plane, a hyperplane) → how good is the fit? ($R^2$) → reading the
coefficients (and the danger of reading too much) → the cliff-hanger: no closed
form in general → Chapter 6 → from-scratch → problems.

**Widgets (3).**
1. `LineFit` — a draggable line over a scatter, the **residuals drawn as vertical
   stubs**, the **MSE** read out live, and a "snap to best fit" button revealing
   the least-squares line — feel the loss shrink as you balance the residuals.
2. `LeastSquares` — the residual *squares* drawn as actual squares whose total
   area *is* the loss; the best-fit line is the one minimising that total area —
   the meaning of "least squares", literally.
3. `ResidualsR2` — the fitted line plus a residual plot and the $R^2$ readout,
   and a noise slider showing $R^2$ fall as the scatter loosens — what "fraction
   of variance explained" looks like.

**Reuse.** `makeNoisyScatter`, `polyFeatures(·, 1)` + `ridgeRegression(·, 0)` or a
tiny inline least-squares (`solveLinear` on the normal equations), `evalPoly`
(`lib/regularisation.ts`). Closes → Chapter 6 (gradient descent), completing the
loop into the already-shipped Part II.

---

## New lib code

Almost none. At most a small `lib/foundations.ts` with:
- `make1DRegression(n, seed, noise)` — a noisy linear/curved 1-D dataset for the
  regression and overfitting widgets (or just reuse `makeNoisyScatter`).
- `leastSquaresLine(xs, ys)` — slope/intercept by the closed form (one-liner; or
  use `ridgeRegression` with degree 1, λ 0).

Everything else reuses shipped libs. Each chapter still gets a Node + Python
numerical pass before its widgets are built (kNN k-effect, the overfitting U,
least-squares = normal-equation solution, $R^2$, etc.), per the project discipline
that has caught every prior issue.

---

## Build order & wiring

Build in narrative order 1 → 2 → 3 → 4 → 5, each: (verify numerically) → widgets →
MDX → wire into `app/page.tsx` (`href`) and `app/sitemap.ts` → `npx next build`
clean → smoke test (HTTP 200, KaTeX, 5/5 problems, no American spellings / no
`Math.random` / no `any` / no backticks-in-Python) → commit + push. One chapter
per commit, as throughout.

Figure numbering per chapter: 1.1–1.2, 2.1–2.3, 3.1–3.3, 4.1–4.3, 5.1–5.3.
Slugs: `1-what-is-machine-learning`, `2-ml-workflow`, `3-mathematical-toolkit`,
`4-k-nearest-neighbours`, `5-linear-regression`.

---

## Open questions for sign-off

1. **Slugs** — the five above (spelled-out, matching the landing titles). Good?
2. **Widget budget** — 2 for Chapter 1, 3 each for Chapters 2–5 (≈14 total),
   lighter than the algorithm chapters since these are foundational and
   prose-led. Happy with that, or want 1 fewer somewhere (e.g. fold Chapter 2's
   `ComplexityCurve` into `OverfitDegree`, or Chapter 5's `LeastSquares` into
   `LineFit`)?
3. **Cadence** — I will build and ship all five sequentially, one commit each,
   with the usual numerical verification, and report at the end (rather than
   pausing for sign-off between each). Assume that unless you would rather review
   chapter by chapter.
```
