# Design plan: Chapter 17 — Feature engineering

> Detailed plan for Chapter 17, written before any code (per the recipe).
> Closes **Part V — More classical models**, and with it the supervised half
> of the book. The one chapter that is not about an algorithm at all but about
> the *input* every algorithm shares: where features come from, and why the
> raw columns you are handed are almost never the columns you should model.

---

## Pedagogical role

Every model in the book so far has taken its features as given — `x` arrived
as a tidy vector and the work was choosing what to do with it. This chapter
pulls the curtain back on that `x`. In practice the representation is a
*choice*, usually the most consequential one in the whole pipeline: the same
data, recolumned well, turns a failing linear model into a winning one, and
the same data recolumned badly defeats the fanciest architecture. "Better
features beat better models" is the closest thing applied machine learning has
to a law, and this is the chapter that earns it.

It is also the great *integrator* of the book. Almost every earlier idea
reappears here as a feature-engineering concern:
- Chapter 4 (kNN) and Chapter 6 (gradient descent): both are wrecked by
  features on mismatched scales — the motivation for **standardisation**.
- Chapter 9 (polynomial ridge) and Chapter 15 (the kernel trick): explicit
  **polynomial and interaction features** are the manual version of the kernel
  trick — feature engineering is lifting the data *by hand* so a linear model
  can curve.
- Chapter 16 (bag of words): **text features** and TF-IDF.
- Chapter 11 (cross-validation): the cardinal sin of **leakage** — fitting a
  transform on data you will later test on — and why every transform must live
  *inside* the CV fold.
- Chapter 9 (L1 sparsity): **feature selection**.

It sets up Part VI. The next chapters turn to *unsupervised* learning, and one
of them — PCA (Chapter 20) — is itself a feature-engineering method: a learned,
automatic transformation of the columns. Chapter 17 is the hand-crafted
prelude to that learned version, and the bridge from supervised to
unsupervised.

Because feature engineering is a craft and a survey rather than a single
algorithm, the chapter leans a little more on prose and breadth than the
algorithm chapters, but keeps the four-widget, runnable-code, five-problem
shape.

---

## The conceptual arc (12 sections)

1. **The unglamorous truth.** Sensory hook: two teams enter a competition with
   the *same* data. One spends a month tuning an exotic model on the raw
   columns; the other spends it building features — a ratio here, a date split
   there, a category folded just so — and feeds them to plain logistic
   regression. The second team wins, and it is not close. The model gets the
   trophy; the features did the work. Hook + promise (scaling, basis
   expansion, encodings, the cardinal sin).

2. **A feature is a choice.** Reframe what `x` is. Raw data — a timestamp, a
   postcode, a free-text field, a price in yen next to a count of clicks — is
   not a feature vector; it is the *raw material* for one. Feature engineering
   is the craft of turning that raw material into the numeric matrix a model
   can use, and every column is a decision about what the model is even
   allowed to see. The model can only be as good as the questions its features
   let it ask.

3. **Scale: putting features on equal footing.** The first and most common
   fix. Any model that measures distance (kNN, Chapter 4) or walks a gradient
   (Chapter 6, and everything trained by it) is at the mercy of feature
   *scale*: a feature measured in the thousands silently drowns out one
   measured in fractions, not because it matters more but because its numbers
   are bigger. **Standardisation** (subtract the mean, divide by the standard
   deviation) and min–max scaling put every feature on equal footing. →
   **Widget 1: FeatureScaling**.

4. **Making linear models curve: basis expansion.** The most powerful move in
   the toolkit. A linear model can only draw straight boundaries *in the
   features it is given* — but if you hand it new columns built from the old
   ones ($x_1^2$, $x_2^2$, the interaction $x_1 x_2$), the very same linear
   model draws curves. This is exactly the kernel trick of Chapter 15, done by
   hand and in the open: lift the data into a richer feature space where a
   line suffices, except here you *build the columns yourself* and can see
   them. Polynomial features, interactions, and basis functions are how linear
   models punch above their weight. → **Widget 2: PolynomialFeatures**.

5. **Encoding categories.** Models eat numbers, but much of the world arrives
   as categories — colours, cities, product types. **One-hot encoding** turns
   a $k$-value category into $k$ binary columns (no false ordering, but an
   explosion of columns for high cardinality). **Ordinal encoding** maps to
   integers (compact, but invents an order that may not exist). **Target /
   mean encoding** replaces a category with the average label for that
   category — compact and powerful, but a *leakage trap* (it peeks at the
   label) that must be done with care inside CV. → **Widget 3:
   CategoricalEncoding**.

6. **Cyclical and temporal features.** A favourite, because the naive thing is
   so obviously wrong once you see it. Encode the hour of day as the integer
   0–23 and you tell the model that 23:00 and 00:00 are as far apart as
   possible, when they are adjacent. The fix is to place periodic quantities
   on a *circle* with a sine/cosine pair, $(\sin\frac{2\pi t}{T},
   \cos\frac{2\pi t}{T})$, so midnight and 23:00 sit next to each other. Dates
   also explode into day-of-week, month, is-holiday, days-since — calendar
   structure the raw timestamp hides. → **Widget 4: CyclicalEncoding**.

7. **Taming skew and outliers.** Many real features — incomes, populations,
   counts — are wildly right-skewed, and a handful of huge values dominate
   distances and gradients. A **log transform** (or Box–Cox / Yeo–Johnson)
   pulls the long tail in and turns a lopsided distribution into a roughly
   symmetric one a model can use, often the single highest-leverage transform
   on tabular data. Clipping and winsorising tame outliers without discarding
   rows.

8. **Missing data.** Real datasets have holes, and most models cannot eat a
   blank. **Imputation** fills them — mean/median for the simple case, model-
   based for the careful one — but the fact that a value was *missing* is often
   itself predictive, so a **missingness indicator** column frequently earns
   its keep. The wrong move (drop every row with any gap) can throw away most
   of the data and bias what remains.

9. **Text and other raw signals.** A short callback and extension of Chapter
   16: free text becomes features through the **bag of words**, refined by
   **TF-IDF**, which down-weights words that appear everywhere and up-weights
   the rare, discriminative ones. The same spirit — turn an unstructured
   signal into informative columns — covers images (edges, histograms) and
   audio (spectra), the hand-built precursors to what deep learning later
   learns automatically.

10. **Feature selection.** More features are not always better — irrelevant
    and redundant columns add variance, cost, and overfitting (the curse of
    dimensionality again). Three families: **filter** methods (rank by
    correlation or mutual information, cheap and model-blind), **wrapper**
    methods (search subsets by retraining, accurate and expensive), and
    **embedded** methods (the model selects as it fits — L1 regularisation from
    Chapter 9 driving coefficients to exactly zero). Often the best feature is
    the one you remove.

11. **The cardinal sin: leakage.** The mistake that silently inflates every
    score and then collapses in production. **Leakage** is letting information
    that will not be available at prediction time — most insidiously, anything
    computed using the *test* labels or the *whole* dataset — sneak into the
    features. Fit your scaler, your imputer, your target encoder on the
    **training fold only**, then apply them to the test fold; do it the other
    way and your cross-validation (Chapter 11) lies to you. The rule: every
    transform is part of the model and must be fit inside the fold.

12. **Implementing it, and problems.** A compact NumPy pass through the core
    transforms — standardise, polynomial-expand, one-hot, cyclical-encode,
    target-encode-without-leaking — then five problems.

### Problems (sketch)

1. **Conceptual — features over models, and leakage.** (No code.) Why does
   standardisation matter for kNN and gradient descent but *not* for a
   decision tree? Give a concrete example of leakage and explain exactly how
   it inflates a cross-validation score. What does target encoding buy you and
   what does it risk?
2. **Standardise, and watch kNN change.** On two features with mismatched
   scales, find a query point's nearest neighbours before and after
   standardisation and show the neighbour set (and the predicted class)
   change.
3. **Basis expansion by hand.** Take a non-linearly-separable set (XOR or
   circles), show logistic regression fails on the raw features, then add the
   right polynomial/interaction columns and watch the same linear model
   separate it perfectly — the kernel trick, done manually.
4. **Encode a category three ways.** One-hot, ordinal, and (leakage-free,
   out-of-fold) target encoding of the same categorical column; compare the
   resulting columns and a downstream model's score.
5. **Leakage, caught in the act.** Scale/encode using the *whole* dataset
   versus the *training fold only*, and show the first gives an optimistic,
   lying cross-validation score. The capstone tying Chapter 11 to this one.

---

## The four visualisations (detailed)

All new, in a new `lib/features.ts` plus a tiny kNN helper. Reuse the class
palette, the heatmap and contour patterns, and existing datasets where they
fit (`makeXOR`, `makeCircles2`, `makeBlobs2`).

### Widget 1 — `FeatureScaling` (build first)

- A two-class set whose two features live on wildly different scales (say
  $x_1 \in [0, 1000]$, $x_2 \in [0, 1]$), classified by **k-nearest
  neighbours**. The decision surface is shown as a heatmap. **Raw:** distance
  is dominated by $x_1$, so the boundary is essentially vertical — $x_2$ is
  invisible to the model. Flip a **standardise** toggle and the boundary
  becomes sensible, both features finally heard. A query point with its $k$
  nearest neighbours (circled) makes the distortion concrete: the "nearest"
  points change completely.
- Pedagogical job: scale is not cosmetic — for any distance- or
  gradient-based model it decides which features the model can even see.
  Callback to Chapters 4 and 6.

### Widget 2 — `PolynomialFeatures` (the manual kernel trick)

- A non-linearly-separable 2-class dataset (XOR-style checkerboard, where the
  interaction term is the key; or concentric circles, where the squares are).
  A row of **checkboxes** for candidate features — $x_1$, $x_2$, $x_1^2$,
  $x_2^2$, $x_1 x_2$ — and a logistic regression fit live on whatever subset is
  ticked, its boundary drawn over the points. With only the raw features the
  boundary is a useless straight line; tick the interaction (or the squares)
  and the *same linear model* suddenly draws the curve that separates the
  classes. The feature matrix is shown growing as columns are added.
- Pedagogical job: feature engineering is the kernel trick made explicit and
  visible — lift the data into columns where a line works. Direct callback to
  Chapters 9 and 15.

### Widget 3 — `CategoricalEncoding`

- A small categorical feature (say six cities) each with a different true
  spam/buy rate. Three side-by-side views of the *same* column: **one-hot**
  (six binary columns — show the sparse 0/1 matrix), **ordinal** (one integer
  column — and a note on the false order it implies), and **target encoding**
  (one column = the category's mean label — a bar chart of the learned rates).
  A toggle reveals the **leakage** danger: target-encoding on all the data
  versus out-of-fold, and how the in-sample version looks suspiciously perfect.
- Pedagogical job: categories must become numbers, and *how* you do it changes
  the column count, the implied structure, and the leakage risk. The most
  practical, least-taught widget.

### Widget 4 — `CyclicalEncoding` (the elegant one)

- Hour-of-day driving a periodic target (busy at midnight, quiet at noon).
  **Left:** the hour as a raw number on a line, with 23:00 and 00:00 at
  opposite ends — and a model that, seeing them as maximally distant, cannot
  learn the wrap-around. **Right:** the same hours mapped by $(\sin, \cos)$
  onto a **clock-face circle**, 23:00 now adjacent to 00:00, the periodic
  pattern suddenly learnable. A slider sweeps the hour and lights up both
  representations together.
- Pedagogical job: the encoding *is* the information. A periodic quantity on a
  line lies about its own geometry; on a circle it tells the truth. The most
  visually satisfying idea in the chapter.

---

## Files to create

```
lib/
  features.ts                   ← NEW: standardise / minmax, polynomialExpand,
                                  cyclicalEncode, one-hot / ordinal / target
                                  encode, a small kNN classifier, and the
                                  synthetic datasets the widgets need.
components/
  FeatureScaling.tsx            ← Widget 1
  PolynomialFeatures.tsx        ← Widget 2
  CategoricalEncoding.tsx       ← Widget 3
  CyclicalEncoding.tsx          ← Widget 4
app/chapters/
  17-feature-engineering/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/17-feature-engineering'`) and
`app/sitemap.ts`. **Proposed slug: `17-feature-engineering`** (matches the
landing title "Feature engineering").

Reused: `makeXOR` / `makeCircles2` / `makeBlobs2` and the `Point` type, the
logistic-regression trainer (`lib/logistic.ts`) for Widget 2, the class
palette and heatmap pattern. No new dependencies.

---

## Library design sketch

```ts
// ── lib/features.ts — new ─────────────────────────────────────────────
import type { Point } from './tree'

// Scaling ────────────────────────────────────────────────────────────
export type Scaler = { mean: number[]; std: number[] }
export function fitScaler(rows: number[][]): Scaler
export function applyScaler(s: Scaler, x: number[]): number[]

// Basis expansion ────────────────────────────────────────────────────
export type PolyTerm = 'x1' | 'x2' | 'x1^2' | 'x2^2' | 'x1*x2'
export function expand(x: [number, number], terms: PolyTerm[]): number[]

// Cyclical encoding ──────────────────────────────────────────────────
export function cyclical(t: number, period: number): [number, number]   // (sin, cos)

// Categorical encoding ───────────────────────────────────────────────
export function oneHot(value: number, k: number): number[]
export function targetEncode(values: number[], labels: number[], k: number): number[]  // category -> mean label

// A small kNN classifier for Widget 1 / Problem 2 ────────────────────
export function knnPredict(train: Point[], x: number[], k: number, scaler?: Scaler): number
export function knnNeighbours(train: Point[], x: number[], k: number, scaler?: Scaler): number[]  // indices

// Synthetic datasets ─────────────────────────────────────────────────
export function makeScaleData(n: number, seed: number): Point[]   // x1 in [0,1000], x2 in [0,1]
export function makeHourlyData(n: number, seed: number): { hour: number; y: number }[]
export const CITY_RATES: { name: string; rate: number }[]         // for the categorical widget
```

Notes & numerical-verification plan (run in Node *before* building widgets):
- **Scaling / kNN (Widget 1):** confirm that on `makeScaleData` raw kNN is
  dominated by $x_1$ (boundary ≈ vertical, low accuracy on the $x_2$-driven
  part) and standardised kNN recovers a sensible boundary and higher accuracy;
  confirm `knnNeighbours` returns a different neighbour set before vs after
  scaling.
- **Basis expansion (Widget 2):** confirm logistic regression on raw XOR is at
  chance (~50%) and that adding the $x_1 x_2$ term lifts it to ~100%; likewise
  circles with the squared terms. Reuse `trainLogistic` on the expanded
  columns.
- **Target encoding / leakage (Widget 3):** confirm in-sample target encoding
  inflates a downstream score versus an out-of-fold encoding, quantifying the
  leakage the widget illustrates.
- **Cyclical (Widget 4):** confirm a model on the raw hour scores worse than on
  the $(\sin, \cos)$ pair for a wrap-around target — the numeric backing for
  the picture.
- **Performance:** all transforms are trivial; kNN over a few hundred points on
  a coarse heatmap grid is the only cost — verify it stays well under a frame.
- **Determinism:** seeded RNG, hand-authored category rates — no `Math.random`,
  no `any`, British spelling throughout.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Class 0 / 1 | `#3c5a8c` / `#c7522a` | all widgets with classes |
| Decision boundary / heatmap | `var(--accent)` + blue↔orange alpha | W1, W2 |
| Query point / neighbours | `var(--ink)` ring | W1 |
| Feature-on / feature-off | `var(--accent)` / `var(--ink-faint)` | W2 checkboxes |
| Encoded columns / bars | class colours + `var(--ink-muted)` | W3 |
| Clock circle / hour marker | `var(--rule)` + `var(--accent)` | W4 |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts and feature
matrices. Figure numbering 17.1–17.4.

---

## Opening paragraph (draft, in voice)

> Two teams are handed the same dataset and the same week. The first team
> treats it as a modelling problem: they reach for the most powerful algorithm
> they know, tune its hyperparameters into the night, stack three models on top
> of one another, and squeeze out every decimal the architecture will give. The
> second team barely changes their model at all — plain logistic regression,
> untouched since the first afternoon — and spends the week on the *columns*
> instead. They split a timestamp into hour-of-day and day-of-week. They take
> the ratio of two numbers that, alone, said nothing. They fold a sprawling
> category down to the one number that mattered. When the scores come in, the
> second team has won, and it is not close.
>
> This is the open secret of applied machine learning: the model gets the
> glory, but the features do the work. Given good features, a humble linear
> model is hard to beat; given bad ones, no architecture on earth will save
> you. Everything in this book so far has quietly assumed the features were
> already there — that `x` arrived as a clean, sensible vector. This chapter is
> about where that vector actually comes from, which is to say, about the most
> consequential and least celebrated craft in the field.
>
> By the end you will know why a feature measured in thousands can silently
> bully one measured in fractions, how to make a straight-line model draw
> curves by handing it the right columns, why encoding the hour of the day as a
> number from 0 to 23 is a small disaster, and the single most expensive
> mistake in all of machine learning — the one that makes your scores look
> wonderful right up until they meet the real world.

## Closing paragraph (draft, in voice)

> Feature engineering is where domain knowledge enters machine learning. The
> algorithms in this book are general — they know nothing about spam, or
> houses, or heartbeats — and it is the features that carry what you understand
> about the problem into a form the algorithm can use. That is why it rewards
> curiosity about the data over cleverness about the model, and why it remains,
> even now, the part of the work that most separates a good practitioner from a
> great one.
>
> It also marks a turning point in the book. With this chapter the supervised
> half is complete: you can take raw data, shape it into features, fit a model
> from a wide repertoire, tune it honestly, and read its errors. What you have
> never done is work *without labels* — and that is where we go next. The
> features you built by hand in this chapter, the next part learns to build and
> compress automatically.
>
> ---
>
> *Next: Chapter 18 — k-means clustering.* Part VI leaves labels behind. When
> no one tells you the answers, can a model still find the structure hiding in
> the data — the groups, the patterns, the hidden axes? Unsupervised learning
> begins.

---

## Expected scope

- One new lib (`lib/features.ts`, ~160 lines: scalers, expansion, encodings,
  cyclical, a small kNN, synthetic datasets) + four components.
- One MDX file (~850 lines, 12 sections, 5 problems). Slightly more prose-heavy
  than the algorithm chapters — feature engineering is a survey — but the same
  shape.
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 `<details>`, 5 "Show solution", sitemap entry, no American
  spellings, no `Math.random`, no `any`).
- Bundle size: ~11–13 kB / ~270 kB First Load JS, in line with Ch 15/16.
- **Numerical verification first:** a Node port confirming the scaling/kNN
  effect, the basis-expansion lift on XOR/circles, the target-encoding leakage
  inflation, and the cyclical-vs-raw-hour gap — before any component is built.

Total estimated work: comparable to Chapter 16 — four self-contained widgets,
each illustrating one transform, with heavy reuse of existing datasets and the
logistic trainer.

---

## Open questions for sign-off

1. **Slug** — `17-feature-engineering` (matches the landing title). Good?
2. **The four widgets** — FeatureScaling (kNN + standardisation) →
   PolynomialFeatures (the manual kernel trick) → CategoricalEncoding (one-hot
   / ordinal / target + leakage) → CyclicalEncoding (the clock). Happy with
   that set? The most likely swap: replacing CategoricalEncoding with a
   dedicated **LeakageDemo** (in-fold vs whole-data transform inflating a CV
   score) — leakage is the chapter's most important warning, though harder to
   make visual. I lean on keeping CategoricalEncoding as a widget and covering
   leakage hard in prose + Problem 5; flag if you'd rather it be a widget.
3. **Polynomial-features dataset** — XOR/checkerboard (so the *interaction*
   term $x_1 x_2$ is the hero) or concentric circles (so the *squared* terms
   are). I lean XOR — the interaction term is the more surprising, less
   Chapter-15-ish reveal — but will take either. Assume XOR unless you object.
```
