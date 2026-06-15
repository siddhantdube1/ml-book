import { createRng, gauss } from './rng'

/**
 * Numerically stable sigmoid. The naive `1 / (1 + Math.exp(-z))` overflows
 * for very negative z, returning NaN. The branched form below stays in the
 * representable range for any finite z.
 */
export function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z)
    return 1 / (1 + e)
  } else {
    const e = Math.exp(z)
    return e / (1 + e)
  }
}

export type LabeledPoint = {
  x: [number, number] // 2D feature vector
  y: 0 | 1 // class label
}

export type LRModel = {
  w: [number, number] // weight vector (2D)
  b: number // bias
}

// ─── Predictions ──────────────────────────────────────────────────────

export function predictProb(model: LRModel, x: [number, number]): number {
  const z = model.b + model.w[0] * x[0] + model.w[1] * x[1]
  return sigmoid(z)
}

export function predict(
  model: LRModel,
  x: [number, number],
  threshold = 0.5,
): 0 | 1 {
  return predictProb(model, x) >= threshold ? 1 : 0
}

// ─── Loss ────────────────────────────────────────────────────────────

const EPS = 1e-12

export function crossEntropy(samples: LabeledPoint[], model: LRModel): number {
  let total = 0
  for (const s of samples) {
    const p = predictProb(model, s.x)
    total += -(s.y * Math.log(p + EPS) + (1 - s.y) * Math.log(1 - p + EPS))
  }
  return total / samples.length
}

export function accuracy(
  samples: LabeledPoint[],
  model: LRModel,
  threshold = 0.5,
): number {
  let correct = 0
  for (const s of samples) {
    if (predict(model, s.x, threshold) === s.y) correct++
  }
  return correct / samples.length
}

// ─── Training ────────────────────────────────────────────────────────

export type LRFrame = {
  model: LRModel
  loss: number
  accuracy: number
  step: number
}

export type TrainOptions = {
  lr: number
  maxSteps: number
  /** Initial weights (default: zeros). */
  w0?: [number, number]
  b0?: number
  /** Stop when gradient norm drops below this. */
  tol?: number
}

/**
 * Run full-batch gradient descent on binary cross-entropy. The gradient
 * has the famously clean form: ∇_w L = X^T (σ(Xw + b) - y) / n.
 * Returns a complete frame history (one entry per step) suitable for
 * scrubbable playback.
 */
export function trainLogistic(
  samples: LabeledPoint[],
  opts: TrainOptions,
): LRFrame[] {
  const { lr, maxSteps, tol = 1e-5 } = opts
  let w: [number, number] = opts.w0 ? [opts.w0[0], opts.w0[1]] : [0, 0]
  let b = opts.b0 ?? 0

  const frames: LRFrame[] = []
  const n = samples.length

  for (let step = 0; step <= maxSteps; step++) {
    const model: LRModel = { w: [w[0], w[1]], b }
    frames.push({
      model,
      loss: crossEntropy(samples, model),
      accuracy: accuracy(samples, model),
      step,
    })

    // Gradient
    let gw0 = 0
    let gw1 = 0
    let gb = 0
    for (const s of samples) {
      const p = predictProb({ w, b }, s.x)
      const err = p - s.y
      gw0 += err * s.x[0]
      gw1 += err * s.x[1]
      gb += err
    }
    gw0 /= n
    gw1 /= n
    gb /= n

    const gNorm = Math.hypot(gw0, gw1, gb)
    if (gNorm < tol) break

    w[0] -= lr * gw0
    w[1] -= lr * gw1
    b -= lr * gb
  }

  return frames
}

// ─── Labeled-data generators ─────────────────────────────────────────
// All generators return points in math coordinates (centred near origin).
// Visualisation components handle the mapping to screen space.

/** Two well-separated Gaussian clusters. Linearly separable. */
export function makeBlobs(
  n: number,
  seed: number,
  separation = 2.5,
  std = 0.7,
): LabeledPoint[] {
  const rng = createRng(seed)
  const out: LabeledPoint[] = []
  for (let i = 0; i < n; i++) {
    const y: 0 | 1 = i % 2 === 0 ? 0 : 1
    const cx = y === 0 ? -separation / 2 : separation / 2
    out.push({
      x: [gauss(rng, cx, std), gauss(rng, 0.3 * (y === 0 ? 1 : -1), std)],
      y,
    })
  }
  // Shuffle so the points aren't strictly alternating
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

/** Two interleaved half-moons. Not linearly separable. */
export function makeMoons(
  n: number,
  seed: number,
  noise = 0.15,
): LabeledPoint[] {
  const rng = createRng(seed)
  const out: LabeledPoint[] = []
  const half = Math.floor(n / 2)
  for (let i = 0; i < half; i++) {
    const t = (i / Math.max(half - 1, 1)) * Math.PI
    out.push({
      x: [
        Math.cos(t) - 0.5 + gauss(rng, 0, noise),
        Math.sin(t) - 0.2 + gauss(rng, 0, noise),
      ],
      y: 0,
    })
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / Math.max(n - half - 1, 1)) * Math.PI
    out.push({
      x: [
        1 - Math.cos(t) - 0.5 + gauss(rng, 0, noise),
        -Math.sin(t) + 0.2 + gauss(rng, 0, noise),
      ],
      y: 1,
    })
  }
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

/** Two clusters with mild overlap — closer than makeBlobs, harder to classify. */
export function makeOverlap(
  n: number,
  seed: number,
): LabeledPoint[] {
  return makeBlobs(n, seed, 1.5, 0.8)
}
