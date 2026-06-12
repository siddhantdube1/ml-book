# Design plan: Chapter 15 — Support vector machines

> Detailed plan for Chapter 15, written before any code (per the recipe).
> Opens **Part V — More classical models**. After two chapters of *ensembles*
> (forests, boosting) that combine many weak rules, this returns to a single,
> deliberately-placed boundary — but the best-placed one: the hyperplane that
> separates the classes with the widest possible margin, and the kernel trick
> that lets that straight boundary curve.

---

## Pedagogical role

Chapter 7 drew a separating line by maximising likelihood; any line that
gets the classes mostly right was acceptable, and logistic regression
settled on one without asking whether it was the *best* line. The support
vector machine asks exactly that question. Of all the hyperplanes that
separate the data, it picks the one that sits as far as possible from the
nearest points of either class — the **maximum-margin** boundary — on the
principle that the most cautious boundary generalises best.

That single idea unfolds into the chapter's three big movements:

1. **Margins and support vectors** — the geometry of "widest gap", and the
   surprise that only a handful of points (the support vectors) determine the
   boundary at all.
2. **Soft margins** — real data overlaps, so we allow violations and pay for
   them, governed by a regularisation knob `C` that is the bias–variance dial
   of Chapters 9–11 in a new costume.
3. **The kernel trick** — lift the data into a higher-dimensional space where
   it *is* linearly separable, and compute the lift implicitly through a
   kernel, so a straight boundary in the lifted space becomes a curved one in
   the original. This is the chapter's showpiece and one of the most elegant
   ideas in the subject.

It compounds on the book:
- Chapter 7 (logistic regression): the comparison case — same linear
  boundary, different objective. The **hinge loss** of the SVM sits beside
  logistic loss, and §6 shows the SVM is *regularised hinge-loss minimisation*
  trainable by the gradient descent of Chapter 6.
- Chapter 9 (regularisation): minimising $\tfrac12\lVert w\rVert^2$ to
  maximise the margin *is* L2 regularisation; `C` is the same margin-vs-error
  trade.
- Chapter 11 (cross-validation): `C` and the kernel width $\gamma$ are
  hyperparameters chosen by exactly the validation machinery already built.
- Chapter 3 (vectors, dot products, projections): the margin is a projection
  computation; this is where that toolkit pays off most visibly.

It sets up the rest of Part V (Naive Bayes, feature engineering) and stands
as the last of the "draw a boundary" models before the book turns to
unsupervised learning and neural networks.

---

## The conceptual arc (12 sections)

1. **The best line.** Sensory hook: a road painted between two rows of
   parked cars. Many lines keep you out of both rows, but only one runs
   exactly down the middle, leaving the most clearance on either side — and
   that is the line you would actually want to drive. Chapter 7 was happy with
   any line that mostly worked; the SVM wants the one with the most room. Hook
   + promise (margins, support vectors, kernels).

2. **Margin and the widest street.** Define the *margin*: the distance from
   the boundary to the nearest training point. A separating hyperplane
   $w\cdot x + b = 0$ has a margin you can widen or narrow by tilting and
   shifting it. The SVM chooses the hyperplane of **maximum margin** — the
   widest "street" you can lay down between the classes without a point
   falling inside it. → **Widget 1: MaxMargin**.

3. **Support vectors.** The defining surprise. The maximum-margin boundary
   touches a few points — the ones sitting exactly on the edge of the
   street — and *only* those points matter. Move or delete any other point and
   the boundary does not budge; move a support vector and it does. These
   points are the *support vectors*, and the model is named for them. A sparse
   summary of the data, and a foretaste of why kernels will be cheap.

4. **The optimisation problem.** Make it precise. Scale $w, b$ so the closest
   points satisfy $y_i(w\cdot x_i + b) = 1$; then the margin is exactly
   $1/\lVert w\rVert$, and maximising it means *minimising* $\tfrac12\lVert
   w\rVert^2$ subject to $y_i(w\cdot x_i+b)\ge 1$ for every point. A clean
   convex (quadratic) program — one global optimum, no local minima. Note the
   tie to Chapter 9: shrinking $\lVert w\rVert$ is L2 regularisation, here
   given a geometric meaning.

5. **Soft margins: when the classes overlap.** Real data is not cleanly
   separable. Introduce *slack* $\xi_i\ge 0$ allowing points to violate the
   margin, and pay a penalty $C\sum_i \xi_i$ for the total violation. Minimise
   $\tfrac12\lVert w\rVert^2 + C\sum_i\xi_i$. The knob `C` trades margin width
   against violations: large `C` punishes every mistake (a narrow, hard margin
   that can overfit), small `C` tolerates mistakes for a wider, smoother
   margin. This is the bias–variance dial again. → **Widget 2: SoftMargin**.

6. **The hinge loss: an SVM is a linear model with a different loss.** The
   reframe that ties the SVM back to Chapters 6–7. The slack penalty is
   exactly the *hinge loss* $\max(0,\,1 - y_i(w\cdot x_i+b))$, so the soft-margin
   SVM is just
   $$\min_w\ \tfrac{\lambda}{2}\lVert w\rVert^2 + \tfrac1n\sum_i \max(0,\,1-y_i(w\cdot x_i+b)),$$
   regularised hinge-loss minimisation — the same shape as regularised
   logistic regression, with hinge in place of log-loss. A small static figure
   compares the two loss curves (hinge's kink at the margin versus logistic's
   smooth tail) and the 0–1 loss both approximate. Consequence: you can train
   a linear SVM by (sub)gradient descent, exactly Chapter 6 — which is what
   §11 does.

7. **The kernel trick (1): lifting to a space where it separates.** The
   showpiece, part one. Some data is hopelessly non-separable by a line — a
   class surrounded by another (concentric rings). But map each 2-D point
   $(x_1,x_2)$ up to 3-D via $(x_1, x_2, x_1^2+x_2^2)$ and the inner ring lifts
   to a bowl below the outer ring: now a flat *plane* slices them apart, and
   projected back down that plane is a *circle*. Non-linear boundaries come
   from linear boundaries in a lifted space. → **Widget 3: KernelLift**.

8. **The kernel trick (2): computing the lift for free.** The genius. The
   SVM's solution depends on the data *only through dot products* $x_i\cdot
   x_j$ (the dual form). A *kernel* $K(x_i,x_j)$ computes the dot product in
   the lifted space *without ever building the lift* — the polynomial kernel
   $(x_i\cdot x_j + 1)^d$ and, above all, the RBF / Gaussian kernel
   $\exp(-\gamma\lVert x_i-x_j\rVert^2)$, which corresponds to an
   infinite-dimensional lift. Swap dot products for $K$ and the linear SVM
   draws arbitrarily curved boundaries at the cost of computing $K$. → **Widget
   4: KernelBoundary**.

9. **Choosing the kernel and its parameters.** The practical handle on §8.
   The RBF width $\gamma$ is a complexity knob just like tree depth: small
   $\gamma$ gives smooth, near-linear boundaries (high bias), large $\gamma$
   wraps tight islands around individual points (high variance, overfit). With
   `C`, that is two hyperparameters, chosen by the cross-validation of Chapter
   11. Rules of thumb: standardise features first (RBF depends on distances);
   try linear first; grid-search $(C,\gamma)$.

10. **Multi-class, and where SVMs sit.** Loose ends. The SVM is inherently
    binary; for $k$ classes use *one-vs-rest* or *one-vs-one* (Chapter 8's
    recipes). Strengths: excellent in high dimensions, effective when
    $p \gg n$ (text), elegant non-linearity via kernels, a sparse
    support-vector solution. Weaknesses: kernel SVMs scale poorly past ~tens
    of thousands of points ($O(n^2)$ kernel matrix), give no native
    probabilities, and need feature scaling. The honest place of the SVM in a
    world that mostly reaches for gradient boosting and neural nets.

11. **Implementing it yourself.** A linear soft-margin SVM by subgradient
    descent (Pegasos), in ~25 lines — the §6 reframe made runnable: at each
    step, shrink $w$ (the regulariser) and, for any point inside the margin,
    nudge $w$ toward classifying it correctly (the hinge subgradient).

12. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual — margins and support vectors.** (No code.) Why does the
   maximum-margin boundary generalise better than an arbitrary separator?
   Which points are support vectors, and why does deleting a non-support
   point leave the boundary unchanged? What does shrinking `C` do to the
   margin and the number of support vectors, and why?
2. **The margin is $1/\lVert w\rVert$.** A short geometric / algebraic
   exercise: given $w,b$ and the constraint $y_i(w\cdot x_i+b)\ge 1$, show the
   distance from the boundary to the nearest point is $1/\lVert w\rVert$, then
   verify it numerically on a tiny separable set.
3. **Hinge versus logistic loss.** Plot $\max(0,1-z)$ and $\log(1+e^{-z})$
   against the margin $z = y(w\cdot x+b)$; confirm both upper-bound 0–1 loss,
   and explain (from the curves) why the SVM's loss yields support vectors —
   zero loss and zero gradient once a point is safely past the margin — while
   logistic regression keeps nudging every point forever.
4. **Train a linear SVM by gradient descent (Pegasos).** Implement the
   subgradient update; train on a separable-ish 2-D set; recover $w,b$, draw
   the boundary and margins, and read off the support vectors (points with
   $y_i(w\cdot x_i+b)\le 1$). Connects §6 and §11.
5. **The kernel trick from scratch.** Take the concentric-rings data, show a
   linear SVM fails, then (a) lift via $(x_1,x_2,x_1^2+x_2^2)$ and separate
   with a linear SVM in 3-D, and (b) confirm an RBF-kernel SVM finds the same
   circular boundary directly. The capstone tying §7–§8 together.

---

## The four visualisations (detailed)

All new, in a new `lib/svm.ts`. Two share a clean separable / mildly-overlapping
blob set; two carry the kernel story (rings + moons). Established palette only.

### Widget 1 — `MaxMargin` (the core; build first)

- Linearly separable two-class blobs. The SVM's maximum-margin boundary
  (solid teal) with its two margin lines (dashed) drawn as a shaded "street",
  and the **support vectors circled**. A slider **tilts a candidate line**
  away from optimal; as it tilts, its own margin band (the distance to its
  nearest point) visibly *narrows*, and a readout compares the candidate's
  margin to the SVM's maximum. A toggle hides the candidate to show the SVM
  solution alone.
- Pedagogical job: "best boundary" = widest margin; the optimum is unique;
  only the support vectors touch the street. The whole premise, made visual.

### Widget 2 — `SoftMargin` (the C knob)

- Two-class blobs with deliberate overlap (not separable). A **`C` slider**
  (log scale) refits the soft-margin SVM: large `C` → a narrow, hard margin
  that contorts to catch nearly every point (few violations, many would-be
  overfit); small `C` → a wide, smooth margin that tolerates several points
  inside it (more support vectors, steadier boundary). The margin street
  widens/narrows live; support vectors and margin-violators are marked; a
  readout shows margin width, number of support vectors, and training
  accuracy.
- Pedagogical job: real data overlaps; `C` trades margin width against
  violations — the bias–variance dial of Chapters 9–11 wearing the SVM's
  clothes.

### Widget 3 — `KernelLift` (the trick made literal; the showpiece)

- **Left panel, 2-D:** concentric rings (inner class 0, outer class 1) — no
  line can separate them; a faint "best linear attempt" boundary is shown
  failing. **Right panel, pseudo-3-D (isometric):** the same points lifted to
  $(x_1, x_2, x_1^2+x_2^2)$ — the inner ring drops into a bowl, the outer ring
  rides high — with a flat separating **plane** sliding between them. A slider
  *raises the lift* (interpolates $z$ from 0 → $x_1^2+x_2^2$), animating the
  rings peeling apart until the plane fits; the plane's shadow back in the
  left panel is the **circular** decision boundary.
- Pedagogical job: non-linear boundaries are linear boundaries in a lifted
  space. The single most illuminating picture in the chapter.

### Widget 4 — `KernelBoundary` (RBF in practice)

- An RBF-kernel SVM on the **moons** (callback to Chapters 13–14, same
  dataset, new model). A probability/decision heatmap with the boundary and
  margins overlaid and support vectors circled. **Two sliders:** $\gamma$
  (kernel width) and `C`. Small $\gamma$ → smooth, almost-linear boundary
  (underfit); large $\gamma$ → tight wiggly islands hugging individual points
  (overfit), the support-vector count climbing. A train/validation-accuracy
  readout makes the sweet spot findable.
- Pedagogical job: the kernel SVM people actually use; $\gamma$ as the
  complexity knob; overfitting and the Chapter-11 tie. Set beside the
  forest/boosting moons boundaries, it completes Part IV–V's tour of how
  different models carve the *same* data.

> A fifth, **static** figure (not interactive) in §6: the hinge-loss vs
> logistic-loss vs 0–1-loss curves, a small inline SVG. Keeps the four
> interactive widgets focused on margins and kernels.

---

## Files to create

```
lib/
  svm.ts                        ← NEW: linear + kernel soft-margin SVM.
                                  Datasets (makeCircles2, reuse makeBlobs2/
                                  makeMoons2), kernels, training, decision fn,
                                  support-vector extraction.
components/
  MaxMargin.tsx                 ← Widget 1
  SoftMargin.tsx                ← Widget 2
  KernelLift.tsx                ← Widget 3
  KernelBoundary.tsx            ← Widget 4
app/chapters/
  15-support-vector-machines/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/15-support-vector-machines'`) and
`app/sitemap.ts`. **Proposed slug: `15-support-vector-machines`** (matches the
landing title "Support vector machines").

Reused: `makeBlobs2` / `makeMoons2` and the `Point` type (`lib/tree.ts`), the
class palette and design tokens, the heatmap + playback patterns from Chapters
13–14, `trainTestSplit` (`lib/crossval.ts`). No new dependencies.

---

## Library design sketch

```ts
// ── lib/svm.ts — new ──────────────────────────────────────────────────
import type { Point } from './tree'   // { x: number[]; y: 0 | 1 }

// Datasets ───────────────────────────────────────────────────────────
export function makeCircles2(n: number, seed: number, noise?: number): Point[]
// inner ring y=0 (small radius), outer ring y=1 (large radius) + gaussian noise.

// Kernels (labels handled as ±1 internally) ──────────────────────────
export type Kernel =
  | { kind: 'linear' }
  | { kind: 'rbf'; gamma: number }
  | { kind: 'poly'; degree: number; coef0: number }
export function kernel(k: Kernel, a: number[], b: number[]): number

// Linear soft-margin SVM via Pegasos (sub-gradient descent on hinge+L2) ─
export type LinearSVM = { w: number[]; b: number }
export type SVMOptions = { C: number; epochs: number; seed: number }
export function trainLinearSVM(data: Point[], opts: SVMOptions): LinearSVM
export function svmDecision(m: LinearSVM, x: number[]): number   // w·x + b
//   margin of model = 1 / ||w||; support vectors: y_i·decision(x_i) ≤ 1 + eps

// Kernel soft-margin SVM via simplified SMO (dual) ───────────────────
export type KernelSVM = {
  alpha: number[]; b: number; sv: Point[]; svY: number[]; kern: Kernel
}
export function trainKernelSVM(data: Point[], kern: Kernel, opts: SVMOptions): KernelSVM
export function kernelDecision(m: KernelSVM, x: number[]): number
//   f(x) = Σ_i alpha_i · y_i · K(sv_i, x) + b ; support vectors are alpha_i > eps
```

Notes & numerical-verification plan (run in Node *before* building widgets):
- **Linear SVM (Widgets 1, 2):** simplified **SMO** (Platt / CS229) on the
  dual gives *exact* support vectors (the points with $\alpha_i>0$), which
  Widgets 1–2 need for clean circling; verify on separable blobs that the
  recovered margin equals $1/\lVert w\rVert$, that exactly the edge points are
  support vectors, and that shrinking `C` widens the margin and grows the SV
  set. Pegasos is the §11 teaching implementation (subgradient, ~25 lines) and
  a cross-check.
- **Kernel SVM (Widgets 3, 4):** the same simplified SMO with a kernel.
  Verify the RBF SVM separates moons (train acc ≈ 1 at moderate $\gamma$),
  that large $\gamma$ overfits (val acc falls, SV count climbs), and that on
  rings the RBF boundary is the circle the lift predicts.
- **Performance:** SMO on ~120 points is an $n^2$ kernel matrix (~14k
  entries) and a few dozen passes — single-digit ms. Train once per
  $(C,\gamma)$, memoised; sliders refit only on release/`useMemo` over the
  slider value. Verify a refit stays well under a frame before shipping.
- **Determinism:** seeded RNG for SMO's point selection and for datasets — no
  `Math.random`, no `any`, British spelling throughout.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Class 0 / 1 | `#3c5a8c` / `#c7522a` | all four widgets |
| Decision boundary | `var(--accent)` (teal) | the separating surface |
| Margin lines / street | `var(--accent)` dashed + faint fill | margins (W1, W2, W4) |
| Support vectors | `var(--ink)` ring | circled points (W1, W2, W4) |
| Candidate / failing line | `var(--ink-muted)` dashed | W1 tilt line, W3 linear attempt |
| Separating plane (3-D) | `var(--accent)` translucent | W3 lifted panel |
| Heatmap | blue→orange alpha by `|decision|` | W4 (the Ch 13–14 scheme) |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure numbering
15.1–15.4 (plus the static hinge-loss figure, unnumbered or 15.x inline).

---

## Opening paragraph (draft, in voice)

> Picture a straight road you have to paint between two rows of parked cars.
> Plenty of lines would keep you clear of both rows — hug the left, hug the
> right, cut a slight diagonal — and a careless painter might choose any of
> them. But there is exactly one line that runs down the true middle, leaving
> the most clearance on either side, and it is obviously the one you would
> want to drive: the line with the most room to be wrong before you clip a
> mirror.
>
> Chapter 7 was the careless painter. Logistic regression drew *a* boundary
> that separated the classes and stopped, never asking whether some other
> boundary sat more safely between them. The support vector machine asks
> precisely that. Of all the lines that separate the two classes, it finds the
> one with the widest *margin* — the largest gap to the nearest point on
> either side — on the wager that the most cautious boundary is the one most
> likely to survive contact with data it has never seen.
>
> That wager leads somewhere remarkable. We will find that only a handful of
> points — the ones pressed right up against the margin — decide the boundary
> at all; that the search for the widest margin is a clean convex problem with
> a single best answer; and that with one of the most elegant manoeuvres in
> the subject, the *kernel trick*, the very same machinery that draws a
> straight line can be made to draw a circle, a pair of arcs, or almost any
> curve you like — without ever leaving the equation for a line.

## Closing paragraph (draft, in voice)

> The support vector machine is the most geometric model in the book. Every
> piece of it is a picture: the widest street between the classes, the
> handful of support vectors holding the boundary in place, the lift into a
> higher dimension where the tangled becomes separable and the plane's shadow
> falls back as a curve. And underneath the geometry sits the familiar
> machinery — a regularised loss, minimised by gradient descent, tuned by
> cross-validation — so the SVM is at once the most visual classical model and
> a close cousin of the linear models we started with.
>
> It is also, in a sense, the end of an era. For two decades the kernel SVM
> was the strongest off-the-shelf classifier in the world. Today gradient
> boosting usually wins on tabular data and neural networks win on
> perception, and the SVM has settled into a specialist's tool — superb in
> high dimensions, on text, when data is scarce. But the ideas it crystallised
> — maximum margins, the kernel trick, sparse support — run straight through
> modern machine learning.
>
> ---
>
> *Next: Chapter 16 — Naive Bayes.* From the most geometric model to the most
> probabilistic: a classifier built almost entirely out of Bayes' rule and a
> single bold independence assumption — and, despite that assumption being
> plainly false, a startlingly strong baseline.

---

## Expected scope

- One new lib (`lib/svm.ts`, ~150 lines: datasets, kernels, SMO, Pegasos,
  decision functions, SV extraction) + four components + one static
  hinge-loss figure (inline SVG in the MDX or a tiny component).
- One MDX file (~850 lines, 12 sections, 5 problems).
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 `<details>`, 5 "Show solution", sitemap entry, no American
  spellings, no `Math.random`, no `any`).
- Bundle size: ~12–14 kB / ~272 kB First Load JS, in line with Ch 13/14.
- **Numerical verification first:** a Node port of `lib/svm.ts` confirming
  margins, support vectors, the `C` and $\gamma$ stories, and the rings-lift
  before any component is built (the discipline that caught the Ch 11/13/14
  issues).

Total estimated work: a touch heavier than Chapter 14 — the SMO solver and
the 3-D lift widget are the two new pieces of real work; everything else
reuses established patterns.

---

## Open questions for sign-off

1. **Slug** — `15-support-vector-machines` (matches the landing title). Good?
2. **The four widgets** — MaxMargin (margins + support vectors) → SoftMargin
   (the `C` knob) → KernelLift (the 2-D→3-D lift showpiece) → KernelBoundary
   (RBF in practice). The kernel story gets two widgets (the literal lift, then
   the practical RBF). Happy with that split, or would you rather, say, drop
   one kernel widget for an interactive **hinge-loss / gradient-descent
   training** widget instead of the static loss figure?
3. **Solver** — I'll implement a simplified **SMO** (dual) as the workhorse so
   support vectors come out exactly (Widgets 1–4 need clean SV circling), and
   present **Pegasos** (subgradient descent) as the §11 "implement it yourself"
   teaching version. Two small solvers rather than one; flagging it as the
   chapter's main engineering. Assume go unless you object.
