# Design plan: Chapter 20 — PCA and dimensionality reduction

> Detailed plan for Chapter 20, written before any code (per the recipe). The
> close of **Part VI — Unsupervised learning**, and the last classical chapter
> before the book turns to neural networks. Where Chapters 18–19 grouped points
> *within* their space, this one reshapes the space itself: it finds the few
> directions along which the data actually varies and throws the rest away —
> the unsupervised, automatic version of the feature engineering you did by hand
> in Chapter 17.

---

## Pedagogical role

Every method so far has taken the data's columns as given and worked in the
space they define. Principal Component Analysis asks a prior question: are these
columns the *right* axes at all? Usually they are not. Real features are
correlated and redundant — height and weight, a dozen sensors reading the same
underlying state, a thousand pixels describing one digit — so the data, though
it lives in a high-dimensional box, actually clusters near a low-dimensional
sheet inside it. PCA finds that sheet: the handful of directions that carry the
data's real variation, ranked, so you can keep the few that matter and discard
the rest.

That one idea pays off three ways the chapter is built around — **visualisation**
(project 100 dimensions down to 2 and finally *see* the structure),
**compression** (store and compute on a few numbers instead of thousands, losing
almost nothing), and **decorrelation/denoising** (the dropped directions are
mostly noise). It is the most widely used unsupervised method in practice, and
the cleanest possible illustration of a theme that runs to the end of the book:
that learning is often about finding a better *representation*.

It compounds on the book:
- Chapter 17 (feature engineering): PCA is feature engineering done
  automatically — it *learns* a new set of features (linear combinations of the
  old) instead of hand-building them, and the closing of Chapter 17 promised
  exactly this.
- Chapter 3 (vectors, projections, the covariance matrix): this is where that
  linear-algebra toolkit is finally spent in full — eigenvectors, projection,
  orthogonality.
- Chapters 18–19 (clustering): PCA's headline visual is projecting
  high-dimensional clustered data to 2-D so the clusters become visible — a
  direct callback, and a reason the two ideas are so often used together.
- Chapter 6 (variance/spread) and Chapter 17 (standardisation): variance is the
  quantity PCA maximises, and scaling decides what "variance" even means.

It sets up Part VII. Neural networks learn their own representations layer by
layer; PCA is the linear, closed-form ancestor of that idea — the gentlest
possible introduction to "the model finds better features than you can".

---

## The conceptual arc (12 sections)

1. **Too many dimensions.** Sensory hook: a spreadsheet a thousand columns wide
   — every pixel of an image, every gene in a sample — that you cannot plot,
   barely store, and which defeats distance-based models (the curse of
   dimensionality, met before). Yet the thousand numbers are not independent:
   they move together. The data only *looks* high-dimensional. Promise: find the
   few directions it actually uses.

2. **Variance is information.** The key reframing. A direction along which the
   data barely varies tells you almost nothing — every point has nearly the same
   value there, so the coordinate is redundant. A direction along which the data
   spreads widely is where the information lives. So "find the important
   directions" becomes "find the directions of maximum variance", and reducing
   dimensions means keeping the high-variance directions and dropping the flat
   ones.

3. **The first principal component.** Make it concrete in 2-D. Among all
   directions you could project the data onto, exactly one captures the most
   variance — the long axis of the cloud. That direction is the **first
   principal component**, PC1. Rotate a candidate axis and watch the spread of
   the projected points swell and shrink; PC1 is where it is widest. →
   **Widget 1: PrincipalAxes**.

4. **The rest of the components.** PC1 alone is a line; to keep more than one
   dimension we need more directions, and the rule is *orthogonality*. PC2 is the
   direction of greatest variance among those *perpendicular* to PC1; PC3
   perpendicular to both; and so on. The principal components are a new,
   rotated set of axes — orthogonal, ordered by how much variance each captures —
   aligned to the data instead of to the arbitrary original columns.

5. **Projection and reconstruction.** Dimensionality reduction is *projecting*
   onto the top components and forgetting the rest. Project a 2-D cloud onto PC1
   and every point collapses to its shadow on that line — a single number per
   point instead of two. Reconstruct by placing each shadow back on the line in
   the original space, and the gap between original and reconstruction is exactly
   the variance you threw away. → **Widget 2: ProjectionReconstruction**.

6. **The mathematics: covariance and eigenvectors.** Where the components come
   from. Centre the data and form the **covariance matrix** $C = \frac1n
   X^\top X$, which records how every pair of features varies together. The
   principal components are its **eigenvectors**, and each one's **eigenvalue**
   is the variance the data has along it. So PCA is one eigendecomposition of one
   symmetric matrix — and, equivalently and more stably, the **singular value
   decomposition** of the centred data, which is how it is actually computed.

7. **How many components to keep?** The eigenvalues answer it. Each component's
   eigenvalue is its variance, and the **explained-variance ratio** is that
   eigenvalue over their sum — the fraction of the data's total spread that
   component accounts for. Plot them in descending order (a **scree plot**) and
   the cumulative curve climbs fast then flattens; keep enough components to
   reach, say, 90–95% of the variance, often a tiny fraction of the original
   dimensions. → **Widget 3: ScreePlot**.

8. **Standardise first.** A practical trap worth its own beat. PCA maximises
   variance, and variance depends on units — measure one feature in millimetres
   and it will dwarf one in metres and hijack PC1, not because it matters more
   but because its numbers are bigger. So, exactly as in Chapter 17, **standardise
   each feature** (mean 0, variance 1) before running PCA, unless the features are
   already in genuinely comparable units. PCA on the covariance matrix versus the
   correlation matrix is precisely this choice.

9. **Seeing high-dimensional data.** The payoff that makes PCA indispensable.
   Take data with real structure hidden in many dimensions — clusters separated
   along directions that no single original feature reveals — and project it onto
   the top two principal components. The structure that was invisible in any pair
   of raw columns snaps into a 2-D picture you can actually look at. →
   **Widget 4: Embedding2D**.

10. **What PCA cannot do.** The honest limits. PCA is **linear** — it can only
    rotate and project, so structure that lives on a curved manifold (a spiral, a
    Swiss roll) defeats it, which is why non-linear methods like **t-SNE** and
    **UMAP** exist for visualisation. Its components are **combinations of all
    features**, so they can be hard to interpret. And it is **unsupervised** —
    it maximises variance, which is not always the same as the directions useful
    for a downstream label. *Complexity*: an eigendecomposition/SVD, roughly
    $O(\min(nd^2, dn^2))$ — cheap for moderate dimensions, and randomised SVD
    handles the rest.

11. **Implementing it yourself.** PCA from scratch in ~20 lines: centre, form
    the covariance matrix, take its eigendecomposition, sort by eigenvalue,
    project onto the top components — then read off the explained variance.

12. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual — variance, axes, and limits.** (No code.) Why does "directions
   of maximum variance" capture "the important directions"? Why must the
   components be orthogonal, and why ordered? Give one kind of structure PCA will
   miss, and say why standardisation can change the result completely.
2. **PCA by hand on a 2-D cloud.** Centre a small correlated dataset, form the
   $2\times2$ covariance matrix, find its eigenvalues/eigenvectors, and confirm
   PC1 is the long axis and its explained-variance ratio matches the picture.
3. **Project and reconstruct.** Project a dataset onto its top-k components and
   reconstruct it; show the reconstruction error equals the sum of the *dropped*
   eigenvalues, and watch it fall as k grows.
4. **Choosing k by explained variance.** On higher-dimensional data, compute the
   explained-variance ratios, plot the cumulative curve, and find the smallest k
   reaching 90% — far below the original dimension.
5. **PCA for visualisation (the clustering tie-in).** Generate clusters in
   high-dimensional space, project to 2-D with PCA, and show the clusters are
   cleanly separated in the PCA view while invisible in any pair of raw
   features — then (optionally) cluster the 2-D projection. The Part-VI capstone.

---

## The four visualisations (detailed)

All new, in a new `lib/pca.ts` (with a small symmetric-matrix eigensolver). The
2-D widgets use a fresh math-coordinate convention; the high-D widgets generate
synthetic data and reduce it. Established palette throughout.

### Widget 1 — `PrincipalAxes` (the max-variance intuition; build first)

- A correlated 2-D cloud with its centroid marked. A **draggable direction
  line** through the centroid; as the reader rotates it, the points' projections
  onto it are shown as ticks on the line and the **projected variance** as a live
  bar. The variance peaks exactly when the line lies along the cloud's long axis —
  and at that angle the line **snaps to PC1**, with PC2 drawn orthogonal. A
  readout gives the angle and the variance captured.
- Pedagogical job: PCA's whole premise in one gesture — of all directions, one
  maximises the spread, and that is the first principal component. The orthogonal
  PC2 makes "the rest, perpendicular" tangible.

### Widget 2 — `ProjectionReconstruction` (lossy compression)

- The same style of 2-D cloud, now with PC1 fixed. A toggle/slider chooses **1 or
  2 components**. At 1 component: each point drops a perpendicular to the PC1
  line (the residual drawn as a faint segment), collapses to its shadow, and the
  reconstruction is the shadow back in 2-D. A readout shows **variance retained**
  ($\lambda_1/(\lambda_1+\lambda_2)$) and **reconstruction error** (the dropped
  $\lambda_2$). At 2 components the reconstruction is exact.
- Pedagogical job: reducing dimension *is* projecting and forgetting; the
  forgotten part is the residual, and its size is exactly the dropped variance —
  lossy compression made visual and quantitative.

### Widget 3 — `ScreePlot` (how many components)

- A higher-dimensional synthetic dataset (say 8-D, with variance concentrated in
  a few directions). A **bar chart of eigenvalues** (variance per component,
  descending) beside the **cumulative explained-variance curve**. A **k slider**
  highlights the first k bars and reads off the cumulative percentage; a 90%
  guide line shows where "enough" lands. The bars fall off a cliff after the few
  real directions — the scree.
- Pedagogical job: the eigenvalue spectrum is the answer to "how many
  components", and the elbow/cliff is the visual signature of the true intrinsic
  dimension.

### Widget 4 — `Embedding2D` (seeing high-dimensional data; the payoff)

- Data with genuine cluster structure generated in **many dimensions** (e.g.
  three clusters whose separating directions are spread across 6–8 features). Two
  panels: **left**, the data plotted on two *raw* features — clusters overlapping,
  structure hidden; **right**, the same data projected onto **PC1 × PC2** — the
  clusters cleanly separated. A control to switch which raw-feature pair the left
  panel shows drives home that *no* pair reveals what PCA does. Points coloured by
  (held-out) cluster id to make the separation legible.
- Pedagogical job: the reason PCA is everywhere — it turns un-seeable
  high-dimensional structure into a picture, and ties the unsupervised part of the
  book together by feeding straight into the clustering of Chapters 18–19.

---

## Files to create

```
lib/
  pca.ts                        ← NEW: centre/standardise, covariance, a Jacobi
                                  eigensolver for symmetric matrices, pca() →
                                  {mean, components, eigenvalues, ratios},
                                  project / reconstruct, and the synthetic
                                  high-D clustered datasets the widgets need.
components/
  PrincipalAxes.tsx             ← Widget 1
  ProjectionReconstruction.tsx  ← Widget 2
  ScreePlot.tsx                 ← Widget 3
  Embedding2D.tsx               ← Widget 4
app/chapters/
  20-pca-dimensionality-reduction/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/20-pca-dimensionality-reduction'`)
and `app/sitemap.ts`. **Proposed slug: `20-pca-dimensionality-reduction`**
(matches the landing title "PCA and dimensionality reduction").

Reused: the `createRng` / `gauss` helpers (`lib/rng.ts`), the class/cluster
palette, the heatmap-free scatter + bar patterns. No new dependencies.

---

## Library design sketch

```ts
// ── lib/pca.ts — new ──────────────────────────────────────────────────

export type PCA = {
  mean: number[]            // per-feature mean (centring)
  components: number[][]    // rows = principal axes, sorted by variance desc
  eigenvalues: number[]     // variance along each component, desc
  ratios: number[]          // eigenvalue / sum (explained-variance ratio)
}

/** Centre (optionally standardise), covariance, eigendecompose, sort. */
export function pca(X: number[][], standardise?: boolean): PCA

/** Project rows of X onto the top-k components → n × k scores. */
export function project(model: PCA, X: number[][], k: number): number[][]
/** Reconstruct from k components back into the original d-space. */
export function reconstruct(model: PCA, X: number[][], k: number): number[][]

/** Jacobi eigendecomposition of a symmetric matrix → sorted (val, vec) pairs. */
export function symmetricEig(A: number[][]): { values: number[]; vectors: number[][] }

// ── Synthetic datasets ───────────────────────────────────────────────
/** A correlated 2-D Gaussian cloud (for Widgets 1–2). */
export function correlatedCloud2D(n: number, seed: number, rho: number): number[][]
/** k clusters in d dimensions, separated along mixed directions (Widgets 3–4). */
export function highDimClusters(n: number, d: number, k: number, seed: number): { X: number[][]; y: number[] }
```

Notes & numerical-verification plan (run in Node *before* building widgets):
- **Eigensolver:** verify `symmetricEig` against known cases (diagonal, a 2×2
  with analytic eigenpairs) and that vectors are orthonormal and values sorted.
- **PCA correctness:** on a correlated 2-D cloud, confirm PC1 lies along the long
  axis (angle matches `0.5·atan2(2·cov_xy, cov_xx−cov_yy)`), eigenvalues sum to
  the total variance, and ratios sum to 1.
- **Reconstruction identity:** confirm reconstruction error (mean squared) equals
  the sum of the dropped eigenvalues, and is 0 when k = d.
- **Embedding:** confirm the high-D clusters are well separated in PC1×PC2
  (e.g. high silhouette / clear gaps) while overlapping in raw feature pairs —
  the contrast Widget 4 depends on.
- **Performance:** Jacobi on ≤8×8 is microseconds; widget datasets ≤300 points.
  Trivial — verify anyway.
- **Determinism:** seeded datasets, deterministic eigensolver — no `Math.random`,
  no `any`, British spelling throughout.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Data points | `var(--ink-muted)` / cluster palette | scatters |
| PC1 axis | `var(--accent)` (teal) | W1, W2 |
| PC2 axis | `var(--ink)` | W1 |
| Candidate direction | `var(--ink-muted)` dashed | W1 |
| Projection residuals | `#c7522a` faint | W2 |
| Reconstructed points | `var(--accent)` | W2 |
| Eigenvalue bars / kept vs dropped | `var(--accent)` / `var(--ink-faint)` | W3 |
| Cumulative-variance curve | `var(--ink)` | W3 |
| Clusters | Ch 18 cluster palette | W4 |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure numbering
20.1–20.4.

---

## Opening paragraph (draft, in voice)

> Open a dataset with a thousand columns — every pixel of a small image, every
> gene in a tissue sample, every sensor on an engine — and you hit a wall
> immediately. You cannot plot it; a scatter needs two axes and you have a
> thousand. You can barely reason about distances in it, because in very high
> dimensions everything is roughly equidistant from everything else. And the
> models that lean on distance, from k-nearest neighbours to k-means, quietly
> fall apart. A thousand numbers per example sounds like a thousand independent
> facts, and the trouble is that it almost never is.
>
> Because those columns are not independent. Pixels next to each other take
> nearly the same value; genes switch on in concert; the engine's twenty sensors
> are all reading off the same few underlying states. The data sprawls across a
> thousand-dimensional box, but it does not fill it — it clusters near a thin,
> low-dimensional sheet folded inside, and almost all of those thousand
> directions are redundant or noise. The real information lives in a handful of
> directions, if only you knew which.
>
> Principal Component Analysis is how you find them. It rotates the data to a new
> set of axes aligned not to your arbitrary original columns but to the data
> itself — ordered so the first axis captures as much of the variation as any
> single direction can, the second as much of what remains, and so on — and then
> lets you keep the first few and discard the rest with almost no loss. It is the
> automatic, learned version of the feature engineering you did by hand in Chapter
> 17, the standard way to compress data and to *see* it, and one of the most used
> ideas in all of machine learning. By the end you will know exactly what those
> axes are, where they come from, and how many of them you need.

## Closing paragraph (draft, in voice)

> Principal Component Analysis is the quiet workhorse of unsupervised learning:
> one eigendecomposition, and a thousand tangled columns become a handful of
> ordered, independent directions you can plot, store, and compute on. It
> compresses, it denoises, it decorrelates, and above all it lets you *see* —
> turning structure that was buried in a hundred dimensions into a picture on a
> page. Its limits are equally clear: it is linear, it is unsupervised, and its
> axes are blends of everything, so when the structure curves or the labels
> matter you reach for other tools. But as a first look at almost any
> high-dimensional dataset, nothing beats it.
>
> It also closes the classical arc of this book. You can now take raw data, shape
> its features, fit and tune supervised models across a wide repertoire, cluster
> without labels, and reduce dimensions to see and compress. Every one of those
> methods, though, drew its boundaries and axes from formulas you could write
> down in closed form or a single greedy loop. The last part asks what happens
> when a model is allowed to *learn its own features*, layer upon layer, with
> nothing but gradient descent and a great deal of data.
>
> ---
>
> *Next: Chapter 21 — The perceptron and multilayer perceptrons.* Part VII begins:
> from a single artificial neuron to a network of them, and the moment machine
> learning starts discovering representations no one designed.

---

## Expected scope

- One new lib (`lib/pca.ts`, ~160 lines: centring/standardising, covariance, a
  Jacobi symmetric eigensolver, pca/project/reconstruct, synthetic datasets) +
  four components.
- One MDX file (~850 lines, 12 sections, 5 problems).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 `<details>`, 5 "Show solution", sitemap entry, no American
  spellings, no `Math.random`, no `any`).
- Bundle size: ~11–14 kB / ~270 kB First Load JS, in line with recent chapters.
- **Numerical verification first:** a Node port confirming the eigensolver, PC1
  along the long axis, eigenvalues = variances summing to the total, the
  reconstruction-error identity, and the high-D-clusters-separate-in-2-D
  contrast — before any component is built.

Total estimated work: comparable to Chapter 19 — the symmetric eigensolver and
the high-D synthetic data are the new pieces; the scatter/bar rendering and
palette carry over.

---

## Open questions for sign-off

1. **Slug** — `20-pca-dimensionality-reduction` (matches the landing title).
   Good?
2. **The four widgets** — PrincipalAxes (max-variance intuition) →
   ProjectionReconstruction (lossy compression) → ScreePlot (how many
   components) → Embedding2D (seeing high-D data). The most likely swap: making
   Widget 4 an **image-reconstruction** demo (a small hard-coded greyscale
   image/digit sharpening as components are added) instead of the
   cluster-embedding. I lean Embedding2D — it ties straight back to Chapters
   18–19 and needs no embedded image data — but will build the image version if
   you prefer the compression angle. Flag your choice.
3. **Standardisation default** — I'll standardise features by default in the
   high-D widgets (and discuss the covariance-vs-correlation choice in §8), but
   leave the 2-D intuition widgets un-standardised so the geometry stays honest
   to the raw cloud. Assume that split unless you object.
```
