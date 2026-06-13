# Design plan: Chapter 21 — The perceptron and multilayer perceptrons

> Detailed plan for Chapter 21, written before any code (per the recipe). The
> opening of **Part VII — Neural networks**, the crown of the book. After
> twenty chapters of models whose boundaries came from a formula or a greedy
> loop, this is where a model is finally allowed to *learn its own features* —
> from a single artificial neuron, to a layer of them, to a network that draws
> any boundary it likes. This chapter builds the *forward* model and the
> intuition; Chapter 22 derives how it learns.

---

## Pedagogical role

Everything in Parts II–VI drew its decision surface from something you could
write down: a line from a closed form (linear/logistic regression), a margin
from a quadratic program (SVM), a partition from greedy splits (trees), a
rotation from an eigendecomposition (PCA). Even the non-linear tricks were
*supplied* — you hand-built the polynomial features (Chapter 17) or chose the
kernel (Chapter 15). The multilayer perceptron is the first model that
manufactures its own non-linear features, automatically, by stacking simple
units — and that single shift is the whole of deep learning in embryo.

The chapter earns the idea from the ground up. A neuron is just a weighted sum
fed through an activation — which, for the right activation, is *exactly* the
logistic regression of Chapter 7, so the reader starts on familiar ground. One
neuron draws one line and, as Chapter 17 already showed with XOR, one line is
not always enough. The fix is to stack neurons into layers: a hidden layer
builds new features, the output layer draws a line *in those features*, and the
combination bends the boundary into any shape. That is the kernel trick and the
basis expansion of earlier chapters, except the features are *learned* rather
than designed — the thread that runs straight to the end of the book.

It compounds on the book:
- Chapter 7 (logistic regression): a single sigmoid neuron *is* logistic
  regression; the network is built by composing them.
- Chapter 6 (gradient descent): the network is trained by the same descent,
  previewed here and derived in Chapter 22.
- Chapter 17 (basis expansion) and Chapter 15 (the kernel trick): hidden layers
  are the *learned* version of manufacturing non-linear features; XOR is the
  shared running example.
- Chapter 20 (PCA / representation): PCA found a better representation in closed
  form; hidden layers find one by learning — the linear ancestor meets its
  non-linear successor.

It sets up Chapter 22, which opens the black box of "and then it learns the
weights": the chain rule, backpropagation, and the practicalities of training.

---

## The conceptual arc (12 sections)

1. **A neuron.** Sensory hook: a single biological neuron, summing signals from
   its dendrites and firing only if the total clears a threshold. Strip it to
   arithmetic — multiply each input by a weight, add a bias, pass the sum
   through an activation — and you have the **artificial neuron**, the atom of
   every network in this book. Promise: from this one atom, any boundary at all.

2. **The perceptron.** The first and simplest neuron model: weighted sum
   $z = w\cdot x + b$ through a hard **step** — fire (1) if $z \ge 0$, else 0. A
   single perceptron is therefore a *linear classifier*; its weights are the
   normal to a separating line, and changing them tilts and shifts that line.
   The reader should feel the déjà vu: swap the step for a sigmoid and this is
   Chapter 7's logistic regression exactly. → **Widget 1: SingleNeuron**.

3. **The perceptron learning rule.** How a perceptron finds its line, and the
   first true *learning* algorithm in the book. Show it a point; if it
   classifies correctly, do nothing; if it errs, nudge the weights toward the
   right answer, $w \leftarrow w + \eta\,(y - \hat y)\,x$. Repeat. On linearly
   separable data this provably converges to a separating line in finite steps.
   → **Widget 2: PerceptronLearning**.

4. **The wall: XOR.** The famous limit that nearly killed the field. A single
   perceptron can only draw a *straight* boundary, and the XOR pattern — the
   checkerboard of Chapter 17 — has none. Watch the learning rule thrash
   forever, never settling, because no line exists to be found. One neuron is
   not enough, and for a decade people thought that was the end. It was the
   beginning.

5. **Stacking neurons: the multilayer perceptron.** The escape. Put several
   neurons in a **hidden layer**, each computing its own weighted sum of the
   inputs, then feed *their* outputs into a final output neuron. The hidden
   units become new, intermediate features; the output neuron draws a line in
   the space those features define. Layers, weights, the shape of a network —
   the vocabulary of every model to come.

6. **Why non-linearity is the whole point.** The subtlety that makes it work. A
   stack of *linear* neurons collapses: a linear function of a linear function
   is still linear, so without something non-linear between the layers, a
   ten-layer network is no more expressive than one. The **activation function**
   is that non-linearity, and the choice matters — the saturating **sigmoid**
   and **tanh**, and the modern default **ReLU** and its leaky cousin, each with
   different shapes and gradients. → **Widget 3: ActivationZoo**.

7. **Forward propagation.** The forward pass, written cleanly. Layer by layer,
   compute $a^{(l)} = \phi(W^{(l)} a^{(l-1)} + b^{(l)})$ — a matrix multiply, an
   added bias, an activation — feeding each layer's output into the next until
   the network emits its prediction. This is all a trained network does to make
   a prediction, and it is just a chain of these three operations.

8. **Solving XOR, and any shape.** The payoff. A tiny network — two inputs, a
   couple of hidden units, one output — cracks XOR that no single neuron could,
   and a wider hidden layer bends the boundary around moons, rings, anything.
   Each hidden unit contributes one "fold" of the boundary; add units and the
   boundary grows as intricate as you like. → **Widget 4: MLPBoundary**.

9. **Hidden layers learn features.** The deep idea, made visual and tied to the
   whole book. The hidden layer is a learned *change of representation*: it
   warps the input space so that, in the hidden units' coordinates, the classes
   become linearly separable and the output neuron's job is trivial. This is the
   kernel trick of Chapter 15 and the basis expansion of Chapter 17 — except the
   network *discovers* the lift instead of being handed it, and that is exactly
   what makes deep learning deep. → **Widget 5: HiddenSpace**.

10. **How big a network? Universal approximation, width, and depth.** The
    theory and the trade-offs, briefly. The **universal approximation theorem**:
    a network with a single hidden layer, given enough units, can approximate
    *any* continuous function to any accuracy — so expressiveness is never the
    obstacle. But "enough" can be astronomically many for a shallow net, whereas
    **depth** — stacking layers — builds complex features from simpler ones far
    more efficiently, which is why modern networks are deep. *Complexity*: a
    forward pass is a few matrix multiplies, $O(\text{number of weights})$ per
    example.

11. **Implementing it yourself.** A two-layer MLP forward pass in NumPy — weights,
    biases, an activation, a matrix multiply per layer — classifying XOR with
    hand-set weights, plus a short peek at training by gradient descent (the
    full mechanics deferred to Chapter 22).

12. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual — neurons, lines, and layers.** (No code.) Why is a single
   perceptron only ever a linear classifier? Why can it not solve XOR, and what
   exactly does adding a hidden layer buy you? Why does a network with *no*
   activation function gain nothing from extra layers?
2. **One neuron is logistic regression.** Implement a single sigmoid neuron's
   forward pass and show it computes the same thing as Chapter 7's logistic
   regression — same weights, same boundary, same probabilities.
3. **The perceptron learning rule.** Implement the rule, train it on a linearly
   separable set (watch it converge), then on XOR (watch it never converge), and
   explain the difference.
4. **Forward pass of an MLP on XOR.** With hand-chosen weights for a 2-2-1
   network, run the forward pass and verify it classifies all four XOR points
   correctly — then perturb a weight and watch a corner flip.
5. **Hidden features make XOR linear.** Push the four XOR points through a
   trained hidden layer and show that in the hidden representation they are
   linearly separable, even though they were not in the input — the learned
   kernel trick, measured.

---

## The five visualisations (detailed)

All new, in a new `lib/neuralnet.ts` (a general MLP: forward pass, activations,
and gradient-descent training via backprop, reused by Chapter 22). The 2-D
widgets use a fresh math-coordinate convention and the class palette.

### Widget 1 — `SingleNeuron` (build first)

- One neuron on a 2-D plane: sliders for the two weights $w_1, w_2$ and bias
  $b$, an **activation selector** (step / sigmoid), and the resulting output
  field drawn as a heatmap with the decision line. A small inset shows the
  activation curve, with the point $z = w\cdot x + b$ marked. Moving a weight
  tilts the line; the bias shifts it; switching step→sigmoid softens the
  boundary from a hard edge to a gradient.
- Pedagogical job: a neuron is a weighted sum through an activation, and that is
  a linear classifier — the same object as logistic regression, now named as the
  network's building block.

### Widget 2 — `PerceptronLearning`

- The perceptron learning rule animated. A linearly separable 2-D set; the
  current boundary; **playback** that, step by step, picks a misclassified
  point, nudges the weights, and rotates the line until every point is correct
  and it stops. A dataset toggle to **XOR**, where the same loop visibly thrashes
  — the line lurching back and forth forever, never settling — making the limit
  unmistakable.
- Pedagogical job: the first learning algorithm, its guaranteed convergence on
  separable data, and the hard wall of XOR that motivates everything after.

### Widget 3 — `ActivationZoo`

- The activation functions side by side — **step**, **sigmoid**, **tanh**,
  **ReLU**, **leaky ReLU** — each curve drawn with its formula, and a small
  companion showing its *derivative* (foreshadowing Chapter 22's gradients, and
  why saturating units learn slowly). A toggle demonstrates the collapse: two
  stacked layers with a *linear* activation produce a straight boundary no matter
  how many units, while the same network with ReLU bends — non-linearity is what
  buys depth.
- Pedagogical job: the menu of activations and the single most important fact
  about them — without a non-linearity, layers are pointless.

### Widget 4 — `MLPBoundary` (the payoff)

- A small MLP trained live on a dataset selector (**XOR / moons / circles**),
  with a **hidden-units slider** (1 → ~10). **Playback** animates training
  (epochs) so the boundary visibly forms; the decision surface is a heatmap with
  the data on top, and the **individual hidden units' lines** are overlaid so the
  reader sees each one contribute a fold. With one hidden unit the boundary is
  straight and fails; add units and it wraps the shape. A train-accuracy readout.
- Pedagogical job: stacked neurons draw non-linear boundaries; XOR falls; the
  number of hidden units controls how intricate the boundary can be.

### Widget 5 — `HiddenSpace` (representation; the deep idea)

- Two panels for a 2-input, 2-hidden-unit network trained on XOR (or two
  not-linearly-separable moons). **Left:** the input space, classes not linearly
  separable, with the hidden units' lines. **Right:** the same points plotted in
  the **hidden units' coordinates** $(h_1, h_2)$ — where the classes have been
  warped apart and a single straight line (the output neuron) now separates them
  cleanly. A slider stepping through training shows the representation
  *un-tangling* as the network learns.
- Pedagogical job: a hidden layer is a learned change of representation that
  makes the problem linearly separable — the kernel trick (Chapter 15) and basis
  expansion (Chapter 17), discovered rather than designed. The conceptual
  capstone of the chapter and the bridge to deep learning.

> Five widgets is at the top of the range, in keeping with this being a flagship
> chapter. If trimming is wanted, `HiddenSpace` (Widget 5) is the natural fold —
> its lesson can lean on `MLPBoundary` plus prose — but it is the most
> book-unifying of the five, so the recommendation is to keep all five.

---

## Files to create

```
lib/
  neuralnet.ts                  ← NEW: activations (+ derivatives), a single
                                  neuron, the perceptron learning rule (frames),
                                  a general MLP (forward pass + backprop
                                  gradient-descent training with frame history),
                                  and the 2-D datasets (xor / moons / circles).
components/
  SingleNeuron.tsx              ← Widget 1
  PerceptronLearning.tsx        ← Widget 2
  ActivationZoo.tsx             ← Widget 3
  MLPBoundary.tsx               ← Widget 4
  HiddenSpace.tsx               ← Widget 5
app/chapters/
  21-perceptron-and-mlps/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/21-perceptron-and-mlps'`) and
`app/sitemap.ts`. **Proposed slug: `21-perceptron-and-mlps`** (the landing title
is "The perceptron and multilayer perceptrons").

Reused: `createRng` / `gauss` (`lib/rng.ts`), the class palette and heatmap
pattern, the frame-history playback pattern from earlier chapters. The `makeXOR`
/ `makeMoons2` / `makeCircles2` generators can be reused or re-created in
math-coordinates. No new dependencies.

---

## Library design sketch

```ts
// ── lib/neuralnet.ts — new ────────────────────────────────────────────

export type Activation = 'step' | 'sigmoid' | 'tanh' | 'relu' | 'leaky'
export function activate(kind: Activation, z: number): number
export function activateDeriv(kind: Activation, z: number): number

// ── Single neuron / perceptron ───────────────────────────────────────
export type Point = { x: [number, number]; y: 0 | 1 }
export type Neuron = { w: [number, number]; b: number }
export function neuronOutput(n: Neuron, x: [number, number], act: Activation): number

export type PerceptronFrame = { w: [number, number]; b: number; errors: number; picked: number | null }
export function trainPerceptron(data: Point[], epochs: number, lr: number): PerceptronFrame[]

// ── Multilayer perceptron (forward + backprop training) ──────────────
export type MLP = { W: number[][][]; b: number[][]; act: Activation }
export function mlpForward(net: MLP, x: number[]): { activations: number[][]; output: number }
export type MLPFrame = { net: MLP; loss: number; accuracy: number; epoch: number }
export function trainMLP(
  data: Point[],
  hidden: number[],         // hidden-layer sizes, e.g. [4] or [8,8]
  opts: { epochs: number; lr: number; act: Activation; seed: number },
): MLPFrame[]               // frame history for playback

// hidden-layer activations for the representation widget
export function hiddenActivations(net: MLP, x: number[], layer: number): number[]

// ── Datasets (math coordinates, ~[-3,3]^2) ───────────────────────────
export function makeXORData(n: number, seed: number): Point[]
export function makeMoonsData(n: number, seed: number): Point[]
export function makeCirclesData(n: number, seed: number): Point[]
```

Notes & numerical-verification plan (run in Node *before* building widgets):
- **Perceptron:** verify convergence on a linearly separable set (errors → 0 in
  finite epochs) and non-convergence on XOR (errors never reach 0, weights
  oscillate) — the contrast Widget 2 depends on.
- **Single neuron = logistic regression:** verify a sigmoid neuron with given
  weights reproduces `lib/logistic.ts`'s probabilities and boundary.
- **MLP training:** verify a 2-`[4]`-1 net trained by backprop reaches ~100% on
  XOR, moons, and circles within a modest epoch budget; verify the loss
  decreases monotonically-ish and the boundary is non-linear. Check the gradient
  numerically (finite differences) on a tiny net to confirm backprop is correct.
- **Hidden representation:** verify the four XOR points become *linearly
  separable* in the trained hidden layer's coordinates (a linear classifier on
  the hidden activations hits 100%) while they were not in the input — the
  measurable form of Widget 5 and Problem 5.
- **Performance:** training a small net for a few hundred epochs on ~150 points
  must stay well under a second so playback is built from a precomputed frame
  history; verify before shipping.
- **Determinism:** seeded weight init and datasets, deterministic training — no
  `Math.random`, no `any`, British spelling throughout.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Class 0 / 1 | `#3c5a8c` / `#c7522a` | all 2-D widgets |
| Decision heatmap | blue↔orange alpha by output | W1, W4 |
| Decision boundary / line | `var(--accent)` | W1, W2, W4 |
| Hidden-unit lines | `var(--ink-muted)` | W4, W5 |
| Activation curve / derivative | `var(--accent)` / `var(--ink-muted)` | W3 |
| Network diagram (nodes/edges) | `var(--ink)` / `var(--rule)` | W1, W5 (small schematic) |
| Highlight / current epoch | `var(--accent)` | playback widgets |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure numbering
21.1–21.5.

---

## Opening paragraph (draft, in voice)

> Every model in this book so far has had its cleverness handed to it. Linear
> and logistic regression draw a line because you told them the features; the
> support vector machine bends that line only through a kernel you chose; the
> decision tree splits on columns you supplied; even principal components are
> just a rotation of the axes you already had. When the data needed a curve, it
> was *you* who built the curve — the squared term, the interaction, the kernel,
> the polynomial. The model only ever fit weights to features a human designed.
>
> The artificial neuron is where that changes. It begins almost insultingly
> simple — a caricature of a brain cell that multiplies each of its inputs by a
> weight, adds them up, and fires if the total crosses a threshold — and on its
> own it is nothing you have not seen: a single neuron is a straight line, the
> logistic regression of Chapter 7 wearing biological clothes. The magic is not
> in the neuron. It is in what happens when you stack them. Wire the outputs of
> one layer of neurons into the inputs of another, put a simple non-linearity
> between them, and the network stops merely fitting features and starts
> *building* them — manufacturing, on its own, exactly the curved coordinates that
> you used to have to hand-craft.
>
> That is the whole idea of deep learning, in one sentence, and this chapter is
> where it begins. You will watch a single perceptron learn a line and then
> thrash helplessly against the XOR problem that no line can solve; you will
> stack a handful of neurons into a hidden layer and watch the same problem fall;
> and you will see, in the hidden layer's own coordinates, the network quietly
> reshaping the data until a straight line is all the answer needs. How it learns
> those weights — the chain rule turned into an algorithm — is the next chapter.
> This one is about what it can build.

## Closing paragraph (draft, in voice)

> The multilayer perceptron is the hinge of this book. Behind it lie twenty
> chapters of models that drew their boundaries from a formula, a margin, a
> split, or a rotation — every one of them fitting weights to features that a
> person had to design. Ahead lies everything that calls itself deep learning,
> and all of it is this same move repeated and scaled: stack simple units, put a
> non-linearity between them, and let the network manufacture its own
> representations layer upon layer. The neuron is a humble atom — a weighted sum
> and a bend — but composed in the thousands and the millions it has learned to
> recognise faces, translate languages, and fold proteins.
>
> We have built the forward pass — what a network *is*, and what it can
> represent — but quietly assumed the one thing that makes it useful: that it can
> find the right weights from data. With a million of them tangled across a dozen
> layers, how could you possibly compute which way to nudge each one? The answer
> is one of the most elegant and important algorithms in all of computing, and it
> is nothing more than the chain rule from calculus, applied with great
> care.
>
> ---
>
> *Next: Chapter 22 — Backpropagation and training.* How a network learns: the
> chain rule turned into an algorithm that finds every weight's gradient at once,
> and the practical craft of actually training the thing.

---

## Expected scope

- One new lib (`lib/neuralnet.ts`, ~200 lines: activations, perceptron, a
  general MLP with backprop training and frame history, datasets) + five
  components.
- One MDX file (~900 lines, 12 sections, 5 problems) — a flagship chapter, so a
  touch longer.
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem
  headings, 5 `<details>`, 5 "Show solution", sitemap entry, no American
  spellings, no `Math.random`, no `any`).
- Bundle size: ~13–16 kB / ~270 kB First Load JS (five widgets, in line with
  Chapter 18's five-widget weight).
- **Numerical verification first:** a Node port confirming perceptron
  convergence vs XOR thrash, the single-neuron/logistic equivalence, MLP
  training to ~100% on XOR/moons/circles with a finite-difference gradient check,
  and the hidden-representation-becomes-separable result — before any component.

Total estimated work: the heaviest chapter since Chapter 18 — the MLP engine
(forward + backprop, reused by Chapter 22) and five widgets — but the playback,
heatmap, and dataset patterns all carry over.

---

## Open questions for sign-off

1. **Slug** — `21-perceptron-and-mlps` (matches the landing title "The
   perceptron and multilayer perceptrons"). Good, or prefer
   `21-neural-networks` / `21-perceptron`?
2. **Five widgets or four?** SingleNeuron → PerceptronLearning → ActivationZoo →
   MLPBoundary, plus HiddenSpace (the learned-representation reveal). I lean on
   **all five** — this is a flagship chapter and HiddenSpace is the most
   book-unifying visual — but HiddenSpace is the clean cut to four if you'd
   rather keep it tighter.
3. **Live training vs pre-trained.** For MLPBoundary / HiddenSpace I'll train the
   small networks in-browser (backprop runs in milliseconds on these toy sets)
   and animate the boundary forming over epochs — a visual *preview* of training
   without deriving backprop, which stays Chapter 22's job. Assume go unless you
   object.
```
