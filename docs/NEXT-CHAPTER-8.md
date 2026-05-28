# Design plan: Chapter 8 — Multi-class classification

> Detailed plan for the chapter most likely to be tackled next. Save you a session of upfront thinking.

---

## Pedagogical role

Chapter 8 is the natural follow-on to Chapter 7. It extends binary logistic regression to the k-class case via the softmax function. It is also where the reader learns that *one-vs-rest* is the obvious-but-flawed extension, and that *softmax* is the principled fix.

The chapter must reuse the gradient-descent machinery from Chapter 6 and the cross-entropy story from Chapter 7. The reader should feel the book compounding — chapter 8 isn't a new topic, it's a deepening.

---

## The conceptual arc

§1. **The problem.** What if there are more than two classes? Digit recognition (10 classes), language identification (~7000 languages), product category (thousands). Binary logistic regression handles only two.

§2. **One-vs-rest: the obvious idea.** Train k binary classifiers, one per class. Each says "is it class 0 or not?", "is it class 1 or not?", etc. At prediction time, pick the class whose classifier scored highest.

§3. **Why one-vs-rest is wrong.** Two problems:
   - The k probabilities don't sum to 1 — they aren't a proper distribution over classes.
   - The classifiers are trained independently and don't know about each other; a point can be "high probability" under multiple classes at once.

   These aren't dealbreakers (one-vs-rest works in practice for many problems), but they motivate the more principled approach.

§4. **The softmax function.** A generalization of the sigmoid to k outputs.
   $$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}$$
   Properties: outputs are positive, sum to 1, naturally a probability distribution. When k=2, softmax reduces to sigmoid (after a parameter substitution).

   **First widget: SoftmaxExplorer.** Show 3 input z-values as draggable sliders or bars. Display the softmax outputs as a horizontal stacked bar (always summing to 1). When you drag one z up, see how the others get squeezed. Pedagogical job: build intuition for the redistribution behavior.

§5. **The multinomial model.**
   $$P(y = c \mid x) = \text{softmax}(W x + b)_c$$
   Now $W$ is a $k \times p$ matrix and $b$ is a $k$-vector. Each class gets its own weight vector and bias.

   **Second widget: MultiClassBoundary.** 2D classification with 3 classes (three Gaussian blobs). Show the soft probability map (RGB triangle gradient — red, green, blue regions blending into each other). Decision boundaries form a Voronoi-like partition: three regions meeting at a single point. Interactive: drag class centroids, watch boundaries reform.

§6. **Categorical cross-entropy.** Generalization of binary cross-entropy.
   $$L(W, b) = -\frac{1}{n} \sum_{i=1}^n \log p_{i, y_i}$$
   where $p_{i, y_i}$ is the predicted probability of the true class for example $i$. This is also $-\sum_i \log p_i^{\top} \mathbf{1}_{y_i}$ when labels are one-hot encoded.

§7. **The gradient is still beautifully clean.**
   $$\nabla_{W_c} L = \frac{1}{n} \sum_i (p_{i,c} - \mathbb{1}[y_i = c]) \, x_i$$
   For each class $c$, the weight gradient is the prediction error (probability minus one-hot label) times the input, averaged. The same shape as binary logistic regression's gradient. **Not a coincidence**: this is the natural derivative of cross-entropy with softmax, and the chain-rule clutter cancels just like in §6 of Chapter 7.

   **Third widget: TrainingDynamics3Class.** Like Chapter 7's TrainingDynamics but with three classes. Left panel shows the softmax probability map (RGB triangle) with the three boundary lines moving during training. Right panel shows the categorical cross-entropy descending. Playback synchronized.

§8. **One-vs-rest vs softmax in practice.** Run both on the same 3-class dataset. Different decision regions emerge. Softmax's regions are cleaner; OvR's regions can have ties or gaps. Both reach similar accuracy on easy problems, but softmax is preferred for principled probabilistic outputs.

§9. **Complexity.**
   - Parameters: $k \cdot p$ weights + $k$ biases, vs $p + 1$ for binary.
   - Per-step cost: $O(nkp)$ instead of $O(np)$.
   - Same convexity guarantee as binary LR — softmax cross-entropy is convex in $W, b$.
   - Practical note: for k = 10, k = 100, k = 10000 (e.g. language models), softmax becomes expensive; this motivates *hierarchical softmax* and *sampled softmax* approximations used in NLP/recommendation systems.

§10. **Implementation from scratch.** ~30 lines of NumPy. Generate three Gaussian blobs, one-hot encode labels, run gradient descent on categorical cross-entropy. Use numerically stable softmax (`max-shift` trick).

§11. **Problems.**

1. **Conceptual** — Why do the k probabilities from one-vs-rest not sum to 1? Why does softmax always sum to 1?

2. **Implement a numerically stable softmax.** The naive formula overflows when any $z_i$ is large. Trick: subtract $\max_j z_j$ from all $z$ before exponentiating; the result is mathematically identical but numerically safe.

3. **Implement categorical cross-entropy.** Given softmax probabilities and one-hot labels, compute the average loss.

4. **Train multinomial logistic regression end-to-end** on three blobs. Compare convergence to Chapter 7's binary version.

5. **One-vs-rest vs softmax** on the same dataset. Train both. Compare decision regions and accuracy. Discuss when each is preferred.

---

## Files to create

```
lib/
  multinomial.ts                ← softmax, predict, categoricalCE, trainMultinomial,
                                  labeled 3-class data generators (3 blobs, 3 moons)
components/
  SoftmaxExplorer.tsx           ← 3 z-bars + stacked-bar output, interactive
  MultiClassBoundary.tsx        ← 3-class 2D playground, RGB probability map
  TrainingDynamics3Class.tsx    ← Two panels: 3-class boundary + categorical CE curve
app/chapters/
  8-multi-class-classification/
    page.mdx                    ← Full chapter
```

---

## Library design sketch

```ts
// lib/multinomial.ts

import { createRng, gauss } from './rng'

export type LabeledPoint = { x: [number, number]; y: number }  // y in {0, 1, ..., K-1}

export type MultinomialModel = {
  W: number[][]   // K x P (K classes, P features)
  b: number[]     // K
}

/** Numerically stable softmax with max-shift. */
export function softmax(z: number[]): number[] {
  const max = Math.max(...z)
  const exps = z.map((zi) => Math.exp(zi - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

export function predictProb(model: MultinomialModel, x: number[]): number[] {
  const z = model.W.map((wc, c) =>
    model.b[c] + wc.reduce((s, wci, i) => s + wci * x[i], 0)
  )
  return softmax(z)
}

export function predict(model: MultinomialModel, x: number[]): number {
  const p = predictProb(model, x)
  let argmax = 0
  for (let i = 1; i < p.length; i++) if (p[i] > p[argmax]) argmax = i
  return argmax
}

export function categoricalCE(samples: LabeledPoint[], model: MultinomialModel): number {
  const EPS = 1e-12
  let total = 0
  for (const s of samples) {
    const p = predictProb(model, s.x)
    total += -Math.log(p[s.y] + EPS)
  }
  return total / samples.length
}

export type MNFrame = {
  model: MultinomialModel
  loss: number
  accuracy: number
  step: number
}

export function trainMultinomial(
  samples: LabeledPoint[],
  K: number,
  opts: { lr: number; maxSteps: number },
): MNFrame[] {
  const P = samples[0].x.length
  let W: number[][] = Array.from({ length: K }, () => new Array(P).fill(0))
  let b = new Array(K).fill(0)
  const frames: MNFrame[] = []
  const n = samples.length

  for (let step = 0; step <= opts.maxSteps; step++) {
    const model = { W: W.map((row) => row.slice()), b: b.slice() }
    frames.push({
      model,
      loss: categoricalCE(samples, model),
      accuracy: samples.filter((s) => predict(model, s.x) === s.y).length / n,
      step,
    })

    // Gradient: for class c, ∇_Wc = (1/n) sum_i (p_{i,c} - 1[y_i = c]) x_i
    const gW: number[][] = Array.from({ length: K }, () => new Array(P).fill(0))
    const gb = new Array(K).fill(0)
    for (const s of samples) {
      const p = predictProb({ W, b }, s.x)
      for (let c = 0; c < K; c++) {
        const err = p[c] - (c === s.y ? 1 : 0)
        for (let j = 0; j < P; j++) gW[c][j] += err * s.x[j]
        gb[c] += err
      }
    }
    for (let c = 0; c < K; c++) {
      for (let j = 0; j < P; j++) W[c][j] -= (opts.lr * gW[c][j]) / n
      b[c] -= (opts.lr * gb[c]) / n
    }
  }
  return frames
}

/** Three Gaussian blobs arranged in a triangle. */
export function makeThreeBlobs(n: number, seed: number, std = 0.6): LabeledPoint[] {
  const rng = createRng(seed)
  const centers: [number, number][] = [
    [-1.5, -0.8],
    [1.5, -0.8],
    [0, 1.4],
  ]
  const out: LabeledPoint[] = []
  for (let i = 0; i < n; i++) {
    const y = i % 3
    const [cx, cy] = centers[y]
    out.push({ x: [gauss(rng, cx, std), gauss(rng, cy, std)], y })
  }
  // shuffle
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}
```

---

## Visualization color palette for 3 classes

Borrow Chapter 7's palette and extend:

- **Class 0**: `#3c5a8c` (blue) — same as Ch 7 class 0
- **Class 1**: `#c7522a` (orange) — same as Ch 7 class 1
- **Class 2**: `#5d8a3a` (olive green) — new

For the probability heatmap in `MultiClassBoundary` and `TrainingDynamics3Class`, blend the three class colors weighted by the softmax probabilities:

```ts
function blendColors(probs: number[]): string {
  // probs sums to 1, length 3
  const r = 60 * probs[0] + 199 * probs[1] + 93 * probs[2]
  const g = 90 * probs[0] + 82 * probs[1] + 138 * probs[2]
  const b = 140 * probs[0] + 42 * probs[1] + 58 * probs[2]
  const alpha = 0.4 * Math.max(...probs)  // more saturated where one class dominates
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(2)})`
}
```

The grid heatmap from Chapter 7 carries over directly — just swap in this color function.

---

## Opening paragraph (draft, in voice)

> Two classes is the simplest version of classification, but it isn't the common case. A digit recognizer has ten classes. A language identifier has thousands. A product taxonomy has tens of thousands. The real world rarely sorts cleanly into "yes" and "no" — most often we need to pick one option from a longer list.
>
> The good news is that nearly everything from Chapter 7 transfers. We keep the same linear scoring, we keep the same idea of squashing scores into probabilities, we keep gradient descent. We change one function — sigmoid becomes *softmax* — and the entire machinery falls into place. The gradient is, again, beautifully clean.

---

## Closing paragraph (draft, in voice)

> Multi-class logistic regression is not the end of the story for classification. For thousands or millions of classes, the cost of computing softmax becomes the bottleneck, and approximations like hierarchical softmax (Chapter 20) or sampled softmax (Chapter 22) take over. For data that isn't linearly separable, the same trick from Chapter 7 applies — engineer richer features, or learn them. The next chapter takes a different turn: instead of more classes, we ask what happens when the model is so flexible it can fit anything, including pure noise. That problem is called overfitting, and the chapter that tames it is called regularisation.
>
> ---
>
> *Next: Chapter 9 — Regularisation.* The geometric story of L1 and L2 and why one of them produces sparse models.

---

## Expected scope

- Three new files in `lib/`, `components/` × 3
- One MDX file (~700 lines)
- Wire-up in `app/page.tsx` and `app/sitemap.ts`
- Build verification + smoke test
- Bundle size: ~10 kB / ~270 kB First Load JS (slightly larger than Ch 7 because of the 3D weight tensor in MultinomialModel)

Total estimated work: similar to Chapter 7 — one focused session.
