import { createRng, gauss } from './rng'

/**
 * Numerically stable softmax with max-shift. Subtracting max(z) before
 * exponentiating leaves the result mathematically identical but keeps every
 * exp argument in (-∞, 0], so no overflow.
 */
export function softmax(z: number[]): number[] {
  let max = -Infinity
  for (const zi of z) if (zi > max) max = zi
  const exps = z.map((zi) => Math.exp(zi - max))
  let sum = 0
  for (const e of exps) sum += e
  return exps.map((e) => e / sum)
}

export type LabeledPoint = {
  x: [number, number]
  y: number // class label in {0, 1, ..., K-1}
}

export type MNModel = {
  W: number[][] // K × P (K classes, P features)
  b: number[] // length K
}

// ─── Predictions ──────────────────────────────────────────────────────

export function predictProb(model: MNModel, x: [number, number]): number[] {
  const z = model.W.map(
    (wc, c) => model.b[c] + wc[0] * x[0] + wc[1] * x[1],
  )
  return softmax(z)
}

export function predict(model: MNModel, x: [number, number]): number {
  const p = predictProb(model, x)
  let argmax = 0
  for (let i = 1; i < p.length; i++) if (p[i] > p[argmax]) argmax = i
  return argmax
}

// ─── Loss ────────────────────────────────────────────────────────────

const EPS = 1e-12

export function categoricalCE(
  samples: LabeledPoint[],
  model: MNModel,
): number {
  let total = 0
  for (const s of samples) {
    const p = predictProb(model, s.x)
    total += -Math.log(p[s.y] + EPS)
  }
  return total / samples.length
}

export function accuracy(samples: LabeledPoint[], model: MNModel): number {
  let correct = 0
  for (const s of samples) {
    if (predict(model, s.x) === s.y) correct++
  }
  return correct / samples.length
}

// ─── Training ────────────────────────────────────────────────────────

export type MNFrame = {
  model: MNModel
  loss: number
  accuracy: number
  step: number
}

export type TrainOptions = {
  K: number
  lr: number
  maxSteps: number
  W0?: number[][]
  b0?: number[]
  tol?: number
}

/**
 * Run full-batch gradient descent on categorical cross-entropy with softmax.
 * The gradient for class c is (1/n) Σ (p_{i,c} − 1[y_i = c]) x_i — the same
 * "(prediction − label) × input" shape as binary logistic regression. The
 * chain-rule cancellation that gives this clean form is the same one
 * Chapter 7 exploited; softmax + categorical cross-entropy is the multi-class
 * generalisation of sigmoid + binary cross-entropy.
 */
export function trainMultinomial(
  samples: LabeledPoint[],
  opts: TrainOptions,
): MNFrame[] {
  const { K, lr, maxSteps, tol = 1e-5 } = opts
  const P = 2 // 2D features throughout the book

  const W: number[][] = opts.W0
    ? opts.W0.map((row) => row.slice())
    : Array.from({ length: K }, () => new Array(P).fill(0))
  const b: number[] = opts.b0 ? opts.b0.slice() : new Array(K).fill(0)

  const frames: MNFrame[] = []
  const n = samples.length

  for (let step = 0; step <= maxSteps; step++) {
    const model: MNModel = {
      W: W.map((row) => row.slice()),
      b: b.slice(),
    }
    frames.push({
      model,
      loss: categoricalCE(samples, model),
      accuracy: accuracy(samples, model),
      step,
    })

    // Gradient
    const gW: number[][] = Array.from({ length: K }, () =>
      new Array(P).fill(0),
    )
    const gb = new Array(K).fill(0)
    for (const s of samples) {
      const p = predictProb({ W, b }, s.x)
      for (let c = 0; c < K; c++) {
        const err = p[c] - (c === s.y ? 1 : 0)
        gW[c][0] += err * s.x[0]
        gW[c][1] += err * s.x[1]
        gb[c] += err
      }
    }

    let gNormSq = 0
    for (let c = 0; c < K; c++) {
      gW[c][0] /= n
      gW[c][1] /= n
      gb[c] /= n
      gNormSq += gW[c][0] * gW[c][0] + gW[c][1] * gW[c][1] + gb[c] * gb[c]
    }
    if (Math.sqrt(gNormSq) < tol) break

    for (let c = 0; c < K; c++) {
      W[c][0] -= lr * gW[c][0]
      W[c][1] -= lr * gW[c][1]
      b[c] -= lr * gb[c]
    }
  }

  return frames
}

// ─── Labeled-data generators ─────────────────────────────────────────
// All generators return points in math coordinates centred near the origin.
// Visualisation components handle the mapping to screen space.

/** Three Gaussian blobs arranged in a triangle. Well separated. */
export function makeThreeBlobs(
  n: number,
  seed: number,
  std = 0.55,
): LabeledPoint[] {
  const rng = createRng(seed)
  const centres: [number, number][] = [
    [-1.6, -0.8],
    [1.6, -0.8],
    [0, 1.4],
  ]
  const out: LabeledPoint[] = []
  for (let i = 0; i < n; i++) {
    const y = i % 3
    const [cx, cy] = centres[y]
    out.push({ x: [gauss(rng, cx, std), gauss(rng, cy, std)], y })
  }
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

/** Three Gaussian blobs with wider spread — classes overlap noticeably. */
export function makeThreeOverlap(n: number, seed: number): LabeledPoint[] {
  return makeThreeBlobs(n, seed, 0.95)
}
