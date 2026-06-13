# Design plan: Chapter 22 — Backpropagation and training

> Detailed plan for Chapter 22, written before any code (per the recipe). The
> **final chapter** of the book and the second half of Part VII. Chapter 21
> built the forward pass — what a network *is* and can represent — and left one
> question open: with a million tangled weights, how do you compute which way to
> nudge each one? This chapter answers it. Backpropagation is the chain rule
> from calculus turned into an algorithm, and around it sits the whole craft of
> actually training a network. The MLP engine (forward + backprop) already
> exists from Chapter 21; here we open it up and explain it.

---

## Pedagogical role

This is where the book's two longest threads finally meet. Chapter 6 taught
gradient descent — follow the slope downhill — but waved at the hardest part:
*computing* the gradient when the function is a deep composition of millions of
parameters. Chapter 21 built exactly such a function (the forward pass) and
deferred exactly that question. Backpropagation is the answer to both, and it is
the single algorithm that makes deep learning computationally possible. Without
it, training a large network would mean perturbing each weight one at a time —
billions of forward passes per step, hopeless. Backprop computes *every* weight's
gradient in a single backward sweep, at the cost of one extra forward pass.

The chapter's job is to make backprop feel inevitable rather than magical — it is
*only* the chain rule, applied with bookkeeping — and then to wrap it in the
practical reality of training: learning rates, the bumpy non-convex landscape,
momentum and Adam, the vanishing-gradient problem that the activations of Chapter
21 foreshadowed, and the overfitting and regularisation that Chapters 9, 11, and
14 prepared. It is the capstone that ties the optimisation of Part II, the
evaluation discipline of Part III, the regularisation of Part II, and the
networks of Chapter 21 into one working whole.

It compounds on the book:
- Chapter 6 (gradient descent): backprop *computes the gradient* that descent
  then follows; the learning rate, momentum, and SGD return, now on a network.
- Chapter 3 (the chain rule, derivatives): the mathematical core, finally spent
  in full.
- Chapter 21 (forward pass, activations): the backward pass mirrors the forward
  pass; saturating activations cause vanishing gradients.
- Chapters 9 / 11 / 14 (regularisation, validation, early stopping): networks
  overfit spectacularly, and the same tools — weight decay, early stopping, plus
  the network-native dropout — tame them.

It closes the book: after this, a reader can build a network, understand exactly
how it learns, and train it without fooling themselves — and is equipped to go on
to the specialised architectures (convolutional, recurrent, transformer) that are
all this same machinery, scaled and shaped.

---

## The conceptual arc (12 sections)

1. **The credit-assignment problem.** Sensory hook: a network gets one example
   wrong, producing a single number of error at the output. But the blame is
   shared among millions of weights buried across a dozen layers — which ones,
   and by how much, should change? Computing each weight's effect by trial and
   error (nudge it, re-run, measure) would take one forward pass *per weight*,
   billions per step. We need every gradient at once. Promise: one clever
   backward pass delivers exactly that.

2. **The chain rule, the only idea you need.** A network is a deep *composition*
   of simple functions, and the gradient of a composition is a product of local
   derivatives — that is the chain rule, $\frac{dL}{dx} = \frac{dL}{dg}\cdot
   \frac{dg}{dx}$. Backprop is nothing more than the chain rule applied to the
   network's composition, organised so the shared sub-computations are done once.
   Build the intuition on a two-link chain before the full network.

3. **The forward pass remembers.** Backprop needs the values from the forward
   pass — each layer's weighted sums and activations — so the forward pass is run
   first and its intermediates *stored*. The network is a computational graph;
   the forward pass fills in every node's value, and the backward pass will fill
   in every node's gradient.

4. **Backpropagation: the backward pass.** The algorithm itself, made concrete.
   Start at the output with the error, then sweep *backward* layer by layer: each
   layer receives the gradient of the loss with respect to its output, and turns
   it into (a) the gradient with respect to its weights — which is what we want —
   and (b) the gradient with respect to its *input*, which it hands to the layer
   before. The error flows backward the way the signal flowed forward. →
   **Widget 1: Backprop** (a tiny network; step the forward pass filling in
   activations, then the backward pass filling in gradients, node by node).

5. **The shape of it: forward signal, backward error.** Distil the mechanics
   into the memorable pattern. Three facts run the whole thing: at the output,
   the error is simply prediction minus target (for the losses we use); to send
   the error back through a layer, multiply by the weights (transposed) and by
   the activation's derivative; and a weight's gradient is the backward error at
   its output times the forward signal at its input, $\partial L/\partial w_{ij}
   = \delta_i\, a_j$. Forward signal meets backward error at every weight.

6. **From gradient to step: batch, mini-batch, SGD.** Backprop gives the
   gradient for one example; training averages it. **Batch** gradient descent
   uses all the data per step (smooth, slow); **stochastic** (SGD) uses one
   example (noisy, fast, and the noise can help escape bad spots); **mini-batch**
   — a few dozen at a time — is the universal compromise and what everyone
   actually uses. An *epoch* is one pass through the data.

7. **The learning rate, again — and why it is everything.** The single most
   important knob, now on a real network. Too small and training crawls; too
   large and the loss oscillates or explodes to NaN; just right and it descends
   smoothly. → **Widget 2: LearningRate** (train an MLP live at a chosen rate,
   watching the loss-versus-epoch curve and the decision boundary together — a
   flat crawl, a clean descent, or a diverging spike).

8. **The landscape is not a bowl.** What makes network training genuinely hard,
   and genuinely surprising. Unlike the convex losses of Parts II–III, a
   network's loss surface is wildly **non-convex** — full of local minima,
   saddle points, and flat plateaus — so gradient descent has no guarantee of
   reaching the global best. The surprise, and a deep empirical fact, is that it
   barely matters: in high dimensions good-enough minima are everywhere, and
   descent reliably finds one that generalises. → **Widget 3: LossLandscape**
   (the real loss surface of a tiny network over two of its weights, as a
   contour map with the descent trajectory — visibly bumpy, with more than one
   valley, yet descent still works).

9. **Better steps: momentum and Adam.** Plain descent is slow in the ravines and
   plateaus that fill a network's landscape. **Momentum** accumulates a running
   velocity so the optimiser rolls through small bumps and accelerates down long
   slopes (Chapter 6's idea, now indispensable). **Adam** goes further, giving
   *each weight its own adaptive learning rate* from running estimates of its
   gradient's size — and it is the default optimiser of modern deep learning. →
   **Widget 4: OptimizerRace** (plain SGD vs momentum vs Adam descending the same
   network loss / training curve side by side — Adam pulling ahead).

10. **Why training is hard, and how we cope.** The practical failure modes and
    their fixes, briefly. **Vanishing/exploding gradients**: the backward
    chain multiplies many derivatives, so in a deep net they can shrink to
    nothing or blow up — the flat tails of sigmoid/tanh (Chapter 21) are a prime
    cause, which is why ReLU, careful **initialisation**, and normalisation
    exist. **Overfitting**: a big network can memorise its training set, so the
    regularisation of the book returns — **weight decay** (L2, Chapter 9),
    **early stopping** (Chapter 11/14), and the network-native **dropout**, which
    randomly silences units so none can be relied on. → **Widget 5:
    OverfitRegularise** (training vs validation loss diverging on a too-powerful
    net, and weight decay / early stopping pulling them back together).
    *Complexity*: backprop costs about the same as the forward pass — $O(\text{number
    of weights})$ per example — which is the whole reason it scales.

11. **Implementing it yourself.** Backpropagation from scratch — the forward
    pass storing activations, the backward pass computing every gradient, and a
    gradient-descent loop — training a 2-layer network on XOR until the loss
    falls and it solves the problem one perceptron could not.

12. **Problems.** Five, increasing difficulty (below).

### Problems (sketch)

1. **Conceptual — the chain rule and the landscape.** (No code.) Why is
   computing gradients by perturbing each weight hopeless, and how does one
   backward pass fix it? Why is a network's loss surface non-convex, and why does
   gradient descent work on it anyway? How do saturating activations cause
   vanishing gradients?
2. **Backprop on one neuron, checked by finite differences.** Implement the
   forward and backward pass for a single sigmoid neuron and confirm the
   backprop gradient matches a numerical (finite-difference) gradient.
3. **Backprop for a 2-layer network.** Implement the full backward pass of a
   2-`h`-1 MLP, verify every gradient against finite differences, and confirm a
   training step reduces the loss.
4. **The learning rate.** Train the same network at several learning rates and
   plot loss-versus-epoch: show the crawl, the clean descent, and the divergence,
   and read off a good rate.
5. **Overfitting and weight decay.** Train a deliberately oversized network on a
   small noisy dataset, watch validation loss turn up while training loss keeps
   falling, then add weight decay (and/or early stopping) and show the gap close.
   The capstone tying Chapters 9 and 11 to neural networks.

---

## The five visualisations (detailed)

All build on `lib/neuralnet.ts` from Chapter 21 (extended with a backprop trace,
optimiser variants, weight decay, and validation-loss recording). The class
palette and the contour/heatmap/playback patterns carry over.

### Widget 1 — `Backprop` (the centrepiece; build first)

- A small network drawn as a node-and-edge diagram (2 inputs → 2 hidden → 1
  output) for one training example. **Phase 1 (forward):** step left-to-right,
  each node lighting up with its computed value (weighted sum, then activation).
  **Phase 2 (backward):** step right-to-left, each node lighting up with its
  *gradient* $\delta$, and each edge with the weight-gradient it produces, the
  error visibly flowing back along the same wires. A Play/Step control walks the
  whole forward-then-backward sweep; the numbers are real (from the engine's
  backprop).
- Pedagogical job: backprop is not magic — it is the chain rule flowing backward
  through the same graph the signal flowed forward through, computing every
  gradient in one sweep.

### Widget 2 — `LearningRate`

- An MLP trained **live** on a fixed dataset (moons), with a **learning-rate
  slider** (log scale) and **playback** over epochs. Two linked views: the
  **loss-versus-epoch curve** drawing itself, and the **decision boundary**
  forming. At a tiny rate the loss barely moves (a flat crawl); at a good rate it
  descends smoothly and the boundary snaps into shape; at too large a rate the
  loss oscillates or spikes upward and the boundary thrashes.
- Pedagogical job: the learning rate governs everything about training a network,
  and its three regimes (too small / good / too large) are unmistakable on the
  loss curve.

### Widget 3 — `LossLandscape`

- The **actual loss surface of a tiny network**, sliced over two of its weights,
  drawn as a contour map (reusing the marching-squares contours of Chapter 6).
  The descent **trajectory** is overlaid (click-to-place start, as in Chapter
  6), and — unlike Chapter 6's clean bowls — the surface is visibly **bumpy**,
  with more than one basin and saddle regions, yet the trajectory still settles
  into a good valley. A toggle to show two starts landing in two different minima.
- Pedagogical job: network losses are non-convex — gradient descent is no longer
  guaranteed optimal — but in practice it reliably finds a good-enough minimum,
  the central empirical surprise of deep learning.

### Widget 4 — `OptimizerRace`

- **Plain SGD**, **momentum**, and **Adam** descending the same network loss,
  shown both as **trajectories** on the `LossLandscape` contour and as **loss
  curves** over steps. Momentum rolls through small bumps and accelerates;
  plain SGD crawls and stalls in flat regions; Adam adapts per-weight and reaches
  a low loss fastest. A start-point control and a play button race them together.
- Pedagogical job: the choice of optimiser matters on real landscapes — momentum
  and especially Adam are why modern networks train at all in reasonable time.

### Widget 5 — `OverfitRegularise`

- A **deliberately oversized** MLP on a **small, noisy** dataset. **Training loss**
  and **validation loss** drawn over epochs: the training loss marches toward
  zero while the validation loss bottoms out and **turns upward** — overfitting,
  the Chapter-11/14 U, now on a network. A **weight-decay slider** (and an
  early-stopping marker at the validation minimum) pulls the two curves back
  together; the decision boundary visibly smooths from a jagged memorised shape
  to a sensible one.
- Pedagogical job: networks overfit hard, and the regularisation of the whole
  book — weight decay (Chapter 9), early stopping (Chapters 11/14), and dropout —
  is exactly the cure. The book's regularisation thread, closed on its most
  powerful model.

> Five widgets, matching Chapter 21 — fitting for the closing pair of the book.
> If a trim is wanted, `OptimizerRace` (Widget 4) is the natural fold into
> `LossLandscape` plus prose, but the SGD/momentum/Adam contrast is worth its own
> view.

---

## Files to create

```
lib/
  neuralnet.ts                  ← EXTEND: a backprop trace (per-node forward
                                  values + backward gradients for one example),
                                  optimiser variants (momentum, Adam) and
                                  mini-batching in trainMLP, optional weight
                                  decay, and validation-loss recording in frames.
                                  Plus a helper to evaluate the loss surface over
                                  two chosen weights for the landscape widget.
components/
  Backprop.tsx                  ← Widget 1
  LearningRate.tsx              ← Widget 2  (name collides with Ch 14's
                                  LearningRate — file will be LearningRateNN.tsx)
  LossLandscape.tsx             ← Widget 3
  OptimizerRace.tsx             ← Widget 4
  OverfitRegularise.tsx         ← Widget 5
app/chapters/
  22-backpropagation-and-training/
    page.mdx                    ← Full chapter
```

Wire-up: `app/page.tsx` (`href: '/chapters/22-backpropagation-and-training'`) and
`app/sitemap.ts`. **Proposed slug: `22-backpropagation-and-training`** (matches
the landing title "Backpropagation and training").

> **Name clash:** Chapter 14 already has `components/LearningRate.tsx`. The new
> learning-rate widget will be `components/LearningRateNN.tsx` to avoid a
> collision.

Reused: `lib/neuralnet.ts` and its datasets (`makeMoonsData`, `makeXORData`,
etc.) from Chapter 21; `lib/contours.ts` (marching squares) for the landscape;
the class palette, heatmap, contour, and playback patterns. No new dependencies.

---

## Library design sketch

```ts
// ── lib/neuralnet.ts — extensions ─────────────────────────────────────

// Per-example backprop trace for the Backprop widget.
export type BackpropTrace = {
  z: number[][]          // pre-activation per layer
  a: number[][]          // activation per layer (a[0] = input)
  delta: number[][]      // dL/dz per layer (the backward error)
  gradW: number[][][]    // dL/dW per layer
  gradB: number[][]      // dL/db per layer
  output: number
  loss: number
}
export function backpropTrace(net: MLP, x: number[], y: 0 | 1): BackpropTrace

// Optimisers + regularisation in training.
export type Optimizer = 'sgd' | 'momentum' | 'adam'
export type MLPTrainOptions = {
  epochs: number
  lr: number
  act: Activation
  seed: number
  optimizer?: Optimizer        // default 'sgd' (full-batch, as Ch 21)
  batchSize?: number           // mini-batch; default = full batch
  weightDecay?: number         // L2 penalty; default 0
  valData?: Point[]            // if given, frames also record validation loss
  recordEvery?: number
}
// MLPFrame gains an optional valLoss field.

// Loss surface over two chosen weights, for the landscape widget.
export function lossSurface(
  net: MLP, data: Point[],
  axis1: WeightRef, axis2: WeightRef,     // which two weights to vary
  range: number, grid: number,
): number[][]                              // grid × grid loss values
export type WeightRef = { layer: number; i: number; j: number } // j<0 → bias i
```

Notes & numerical-verification plan (run in Node *before* building widgets):
- **Backprop trace:** re-confirm (finite differences) that `gradW`/`gradB` match
  numerical gradients on a tiny net — the Backprop widget shows these numbers, so
  they must be exactly right.
- **Optimisers:** verify momentum and Adam reduce the loss *faster* than plain
  SGD on moons (fewer epochs to a target loss), and that Adam is robust across a
  range of learning rates — the contrast Widget 4 depends on.
- **Learning-rate regimes:** verify a tiny lr barely moves the loss, a good lr
  descends smoothly to low loss, and a too-large lr makes the loss increase /
  go NaN — the three regimes of Widget 2.
- **Non-convexity:** verify that two different starts (or seeds) on the tiny-net
  loss surface land in *different* minima with different losses — the point of
  Widget 3.
- **Overfitting + weight decay:** verify an oversized net on a small noisy set
  drives training loss down while validation loss turns up, and that a suitable
  weight decay narrows the gap (lower validation loss) — Widget 5.
- **Performance:** all training is on ≤200-point toy sets for a few hundred
  epochs (single-digit to low-tens of ms); landscapes are ≤60×60 grids of cheap
  forward passes. Verify playback is built from precomputed frames.
- **Determinism:** seeded init, datasets, and (seeded) mini-batch shuffling —
  no `Math.random`, no `any`, British spelling throughout.

---

## Visualisation colour palette

Reuse the established tokens — no new colours.

| Role | Colour | Use |
|---|---|---|
| Class 0 / 1 | `#3c5a8c` / `#c7522a` | boundary widgets |
| Forward value / signal | `var(--accent)` | Backprop forward phase |
| Backward gradient / error | `#c7522a` | Backprop backward phase |
| Network nodes / edges | `var(--ink)` / `var(--rule)` | Backprop diagram |
| Loss curve (train) | `var(--accent)` | LearningRate, OverfitRegularise |
| Validation loss | `var(--ink)` / `#c7522a` | OverfitRegularise |
| Contours / trajectory | `var(--rule)` / `var(--accent)` | LossLandscape |
| SGD / momentum / Adam | three palette colours | OptimizerRace |

Fonts: `font-sans` UI/labels, `font-mono` numeric readouts. Figure numbering
22.1–22.5.

---

## Opening paragraph (draft, in voice)

> A trained network is a marvel of cooperation: a million numbers, spread across
> a dozen layers, that together turn a photograph into the word "cat". An
> untrained one is a million numbers chosen at random, and it turns the same
> photograph into noise. Everything that separates the two — all of the
> intelligence — lives in the gap, and getting from one to the other means
> answering a question that sounds impossible. The network makes a mistake; the
> error arrives as a single number at the very end. Which of the million weights,
> buried layers deep, was to blame, and by how much should each one change?
>
> You could imagine finding out by brute force: nudge one weight a hair, run the
> whole network again, and see whether the error went up or down. Do that for
> every weight and you would know the gradient — the direction to step each one —
> exactly. You would also need a million forward passes for a single step of
> learning, and a billion-weight network would never finish its first. Brute
> force is a non-starter by a factor of millions, and for a long time the lack of
> anything better is what kept neural networks a curiosity.
>
> The thing that is better is **backpropagation**, and its secret is almost
> deflating: it is just the chain rule from first-year calculus, applied with
> enough care to share its work. In a single backward sweep through the network —
> one extra pass, not a million — it computes the gradient of the loss with
> respect to *every* weight at once. That one idea is what turned neural networks
> from a curiosity into the most powerful technology of the age, and it is, at
> bottom, a piece of bookkeeping. This chapter derives it, watches it run, and
> then turns to the craft that surrounds it — learning rates, optimisers, bumpy
> landscapes, and the ever-present danger of a network that memorises instead of
> learns.

## Closing paragraph (draft — the book's ending)

> And that is the engine. A network is a tower of weighted sums and gentle bends;
> backpropagation is the chain rule that tells every weight which way to move; and
> gradient descent, step by patient step, walks the whole tower downhill toward
> something that works. Everything you have read about since Chapter 21 — and
> everything you will read about beyond this book, the convolutional networks that
> see, the recurrent and attention-based networks that read and speak — is this
> same machinery, scaled up and shaped to its task. The atom did not change; we
> only learned to build with it.
>
> Look back over the whole arc. You began with a single straight line fit to a
> cloud of points, and the questions that line could not answer pulled you
> forward: how to optimise it when there was no formula, how to bend it, how to
> trust it, how to grow a forest of them, how to find structure with no labels at
> all, and finally how to stack the simplest possible unit into something that
> learns its own way of seeing. Every chapter was a model, but the real subject
> was always the same handful of ideas worn smooth by repetition — a model, a
> loss, a way to make the loss smaller, and the unblinking discipline of asking
> whether it actually generalises. Those ideas do not run out at the end of this
> book. They are the whole of the field, and you now have them. Go build
> something.

---

## Expected scope

- Extend `lib/neuralnet.ts` (backprop trace, optimisers, weight decay,
  validation loss, loss-surface helper; ~120 added lines) + five components.
- One MDX file (~900 lines, 12 sections, 5 problems) — the closing flagship.
- Wire-up in `app/page.tsx` and `app/sitemap.ts`.
- Build verification + smoke test (HTTP 200, KaTeX present, 5 problem headings,
  5 `<details>`, 5 "Show solution", sitemap entry, no American spellings, no
  `Math.random`, no `any`, no backticks in Python comments).
- Bundle size: ~14–17 kB / ~270 kB First Load JS (five widgets).
- **Numerical verification first:** a Node port confirming the backprop-trace
  gradients (finite differences), the optimiser speed ordering, the three
  learning-rate regimes, non-convex multi-minimum landscapes, and the
  overfitting/weight-decay effect — before any component is built.

Total estimated work: on par with Chapter 21 — the engine extensions
(optimisers, trace, regularisation) and five widgets, with the Backprop diagram
as the one genuinely new rendering pattern. The book's last build.

---

## Open questions for sign-off

1. **Slug** — `22-backpropagation-and-training` (matches the landing title
   "Backpropagation and training"). Good, or prefer `22-backpropagation` /
   `22-training`?
2. **Five widgets or four?** Backprop → LearningRate → LossLandscape →
   OptimizerRace → OverfitRegularise. I lean on **all five** (matching Chapter
   21, fitting for the finale); `OptimizerRace` is the natural cut to four if you
   want it tighter, since Chapter 6 already covered momentum/SGD (though Adam and
   the on-a-network framing are new here).
3. **Overlap with Chapter 6.** Some learning-rate / landscape / momentum ground
   was covered generically in Chapter 6. I'll keep Chapter 22's versions firmly
   *network-specific* — loss-versus-epoch curves on a real MLP, the network's own
   non-convex surface, Adam — and cross-reference Chapter 6 rather than repeat it.
   Assume that framing unless you object.
```
