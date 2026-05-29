import { createRng, gauss } from './rng'
import { sigmoid } from './logistic'

// ─── Small dense matrix utilities ────────────────────────────────────
// All matrices are row-major number[][]; rows have equal length. For
// chapter 9 widgets the dimensions are small (≤16), so we don't bother
// with packed storage or BLAS-style routines — clarity wins.

export type Matrix = number[][]

export function matT(A: Matrix): Matrix {
  const m = A.length
  const n = A[0].length
  const T: Matrix = Array.from({ length: n }, () => new Array(m).fill(0))
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j]
  return T
}

export function matMul(A: Matrix, B: Matrix): Matrix {
  const m = A.length
  const k = A[0].length
  const n = B[0].length
  const C: Matrix = Array.from({ length: m }, () => new Array(n).fill(0))
  for (let i = 0; i < m; i++) {
    for (let l = 0; l < k; l++) {
      const a = A[i][l]
      if (a === 0) continue
      for (let j = 0; j < n; j++) C[i][j] += a * B[l][j]
    }
  }
  return C
}

export function matVec(A: Matrix, v: number[]): number[] {
  const m = A.length
  const n = A[0].length
  const r: number[] = new Array(m).fill(0)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += A[i][j] * v[j]
    r[i] = s
  }
  return r
}

/**
 * Solve A x = b by Gauss-Jordan elimination with partial pivoting. A is
 * assumed square and well-conditioned; for ridge regression we always
 * call this on X^T X + λI which is symmetric positive-definite for λ > 0.
 */
export function solveLinear(A: Matrix, b: number[]): number[] {
  const n = A.length
  const M: Matrix = A.map((row, i) => [...row, b[i]])
  for (let i = 0; i < n; i++) {
    let maxRow = i
    let maxVal = Math.abs(M[i][i])
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > maxVal) {
        maxRow = k
        maxVal = Math.abs(M[k][i])
      }
    }
    if (maxRow !== i) {
      const tmp = M[i]
      M[i] = M[maxRow]
      M[maxRow] = tmp
    }
    const pivot = M[i][i]
    if (Math.abs(pivot) < 1e-14) {
      // Numerically singular — return zeros rather than throwing in a render
      // loop. Ridge with λ > 0 should never get here on the inputs we use.
      return new Array(n).fill(0)
    }
    for (let j = i; j <= n; j++) M[i][j] /= pivot
    for (let k = 0; k < n; k++) {
      if (k === i) continue
      const factor = M[k][i]
      if (factor === 0) continue
      for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j]
    }
  }
  return M.map((row) => row[n])
}

// ─── Polynomial features ─────────────────────────────────────────────

/**
 * Build the design matrix [1, x, x², …, x^degree] for each x in xs.
 * Returns an (n × (degree + 1)) matrix.
 */
export function polyFeatures(xs: number[], degree: number): Matrix {
  return xs.map((x) => {
    const row: number[] = new Array(degree + 1)
    let xp = 1
    for (let d = 0; d <= degree; d++) {
      row[d] = xp
      xp *= x
    }
    return row
  })
}

/**
 * Closed-form ridge regression: minimise (1/2n) ‖Xw − y‖² + λ ‖w_{1:}‖².
 * The intercept (column 0, which we assume is the constant feature) is
 * left unregularised — standard convention.
 *
 * Returns the weight vector w. Length matches X[0].length.
 */
export function ridgeRegression(
  X: Matrix,
  y: number[],
  lambda: number,
): number[] {
  const Xt = matT(X)
  const XtX = matMul(Xt, X)
  const p = XtX.length
  // Add λ · I to all diagonal entries except the intercept (column 0).
  for (let j = 0; j < p; j++) XtX[j][j] += j === 0 ? 0 : lambda
  const Xty = matVec(Xt, y)
  return solveLinear(XtX, Xty)
}

/** Evaluate a polynomial Σ w_d x^d at a single x. */
export function evalPoly(w: number[], x: number): number {
  let y = 0
  let xp = 1
  for (let d = 0; d < w.length; d++) {
    y += w[d] * xp
    xp *= x
  }
  return y
}

// ─── 1D regression dataset for the polynomial widget ─────────────────

/**
 * The "true" function the polynomial widget tries to recover. Defined on
 * x ∈ [-1, 1]; centring on the origin keeps the monomial basis well
 * conditioned at high degrees. One full period of the dominant sine plus
 * a sub-harmonic gives the overfitting picture enough shape to be
 * visually interesting.
 */
export function trueCurve(x: number): number {
  return 0.7 * Math.sin(Math.PI * x) + 0.22 * Math.sin(2.5 * Math.PI * x)
}

export type ScatterPoint = { x: number; y: number }

export function makeNoisyScatter(
  n: number,
  seed: number,
  noise = 0.18,
): ScatterPoint[] {
  const rng = createRng(seed)
  const out: ScatterPoint[] = []
  for (let i = 0; i < n; i++) {
    const x = -1 + (2 * i + 1) / n
    out.push({ x, y: trueCurve(x) + gauss(rng, 0, noise) })
  }
  return out
}

// ─── 2D quadratic loss for the constraint-geometry widget ────────────

/**
 * Rotation angle and eigenvalues used to construct the quadratic loss for
 * the constraint-geometry widget. Exported so the widget can draw the
 * exact loss-contour ellipses without re-deriving the Hessian.
 */
export const LOSS_THETA = (25 * Math.PI) / 180
export const LOSS_EIGS: [number, number] = [1, 0.22]

/**
 * A 2D quadratic loss L(w) = ½ (w − w*)^T H (w − w*). H is a rotated,
 * stretched ellipse — chosen to make the L1 corner-touch geometry
 * visually striking rather than for any data-fitting reason.
 */
export const LOSS_HESSIAN: Matrix = (() => {
  const c = Math.cos(LOSS_THETA)
  const s = Math.sin(LOSS_THETA)
  const [d1, d2] = LOSS_EIGS
  const D: Matrix = [
    [d1, 0],
    [0, d2],
  ]
  const R: Matrix = [
    [c, -s],
    [s, c],
  ]
  return matMul(matT(R), matMul(D, R))
})()

export function quadLoss(
  w: [number, number],
  wStar: [number, number],
  H: Matrix = LOSS_HESSIAN,
): number {
  const d0 = w[0] - wStar[0]
  const d1 = w[1] - wStar[1]
  return 0.5 * (H[0][0] * d0 * d0 + 2 * H[0][1] * d0 * d1 + H[1][1] * d1 * d1)
}

/**
 * Numerically find the minimum-loss point on the L2 ball of radius t.
 * Samples N angles. N = 720 gives sub-degree resolution — well within
 * what's perceptible for a single widget update.
 */
export function constrainedOptL2(
  wStar: [number, number],
  t: number,
  H: Matrix = LOSS_HESSIAN,
  N = 720,
): [number, number] {
  // If unconstrained optimum is inside the ball, no constraint binds.
  if (Math.hypot(wStar[0], wStar[1]) <= t) return [wStar[0], wStar[1]]
  let best: [number, number] = [t, 0]
  let bestLoss = Infinity
  for (let k = 0; k < N; k++) {
    const theta = (k / N) * 2 * Math.PI
    const w: [number, number] = [t * Math.cos(theta), t * Math.sin(theta)]
    const L = quadLoss(w, wStar, H)
    if (L < bestLoss) {
      bestLoss = L
      best = w
    }
  }
  return best
}

/** Numerically find the minimum-loss point on the L1 diamond of radius t. */
export function constrainedOptL1(
  wStar: [number, number],
  t: number,
  H: Matrix = LOSS_HESSIAN,
  perEdge = 200,
): [number, number] {
  if (Math.abs(wStar[0]) + Math.abs(wStar[1]) <= t) {
    return [wStar[0], wStar[1]]
  }
  const corners: [number, number][] = [
    [t, 0],
    [0, t],
    [-t, 0],
    [0, -t],
  ]
  let best: [number, number] = corners[0]
  let bestLoss = Infinity
  for (let c = 0; c < 4; c++) {
    const a = corners[c]
    const b = corners[(c + 1) % 4]
    for (let k = 0; k <= perEdge; k++) {
      const u = k / perEdge
      const w: [number, number] = [
        a[0] + (b[0] - a[0]) * u,
        a[1] + (b[1] - a[1]) * u,
      ]
      const L = quadLoss(w, wStar, H)
      if (L < bestLoss) {
        bestLoss = L
        best = w
      }
    }
  }
  return best
}

// ─── Logistic regression with L2 and L1 penalties ───────────────────

export type LRPoint = {
  x: number[] // length p
  y: 0 | 1
}

export type RegFrame = {
  w: number[]
  b: number
  loss: number
  step: number
}

/**
 * Generate the labelled dataset used by the regularisation-path widget.
 * The first `nInformative` features carry signal; the rest are pure noise.
 * The true weight on feature j (j < nInformative) is set via `signalCoefs`.
 */
export function makeSparseDataset(
  n: number,
  nInformative: number,
  nNoise: number,
  signalCoefs: number[],
  seed: number,
  labelNoise = 0.4,
): LRPoint[] {
  const rng = createRng(seed)
  const p = nInformative + nNoise
  const out: LRPoint[] = []
  for (let i = 0; i < n; i++) {
    const x: number[] = new Array(p)
    for (let j = 0; j < p; j++) x[j] = gauss(rng, 0, 1)
    let z = 0
    for (let j = 0; j < nInformative; j++) z += signalCoefs[j] * x[j]
    z += gauss(rng, 0, labelNoise)
    out.push({ x, y: z >= 0 ? 1 : 0 })
  }
  return out
}

function logisticPredictProb(
  w: number[],
  b: number,
  x: number[],
): number {
  let z = b
  for (let j = 0; j < w.length; j++) z += w[j] * x[j]
  return sigmoid(z)
}

function logisticLoss(samples: LRPoint[], w: number[], b: number): number {
  const eps = 1e-12
  let total = 0
  for (const s of samples) {
    const p = logisticPredictProb(w, b, s.x)
    total += -(s.y * Math.log(p + eps) + (1 - s.y) * Math.log(1 - p + eps))
  }
  return total / samples.length
}

/**
 * Train logistic regression with an L2 penalty: minimise
 *   (1/n) Σ BCE(y_i, σ(w · x_i + b)) + λ ‖w‖².
 * The bias is unregularised. Warm-start by passing w0/b0.
 *
 * The L2 penalty adds a term 2λw to the gradient, so the per-step
 * shrinkage factor is (1 − 2·lr·λ). With a fixed learning rate this
 * overshoots and diverges once 2·lr·λ > 1 (large λ). We damp the step
 * size by the penalty curvature — lrEff = lr / (1 + 2·lr·λ) — which keeps
 * 2·lrEff·λ < 1 for every λ while leaving lrEff ≈ lr when λ is small.
 * A divergence guard backs out any non-finite step as a final safety net.
 */
export function trainLogisticL2(
  samples: LRPoint[],
  lambda: number,
  lr: number,
  maxSteps: number,
  w0?: number[],
  b0?: number,
): { w: number[]; b: number; loss: number } {
  const n = samples.length
  const p = samples[0].x.length
  let w: number[] = w0 ? w0.slice() : new Array(p).fill(0)
  let b: number = b0 ?? 0
  const lrEff = lr / (1 + 2 * lambda * lr)

  for (let step = 0; step < maxSteps; step++) {
    const gw: number[] = new Array(p).fill(0)
    let gb = 0
    for (const s of samples) {
      const prob = logisticPredictProb(w, b, s.x)
      const err = prob - s.y
      for (let j = 0; j < p; j++) gw[j] += err * s.x[j]
      gb += err
    }
    const wNext = new Array(p)
    for (let j = 0; j < p; j++) {
      wNext[j] = w[j] - lrEff * (gw[j] / n + 2 * lambda * w[j])
    }
    const bNext = b - lrEff * (gb / n)
    if (!Number.isFinite(bNext) || wNext.some((v) => !Number.isFinite(v))) {
      break // diverged — keep the last finite iterate
    }
    w = wNext
    b = bNext
  }
  return { w, b, loss: logisticLoss(samples, w, b) }
}

/** soft-threshold: sign(z) · max(|z| − t, 0). */
function softThreshold(z: number, t: number): number {
  if (z > t) return z - t
  if (z < -t) return z + t
  return 0
}

/**
 * Train logistic regression with an L1 penalty via proximal gradient
 * (ISTA): take a gradient step on the smooth part, then soft-threshold.
 * Bias is unregularised. Warm-start by passing w0/b0.
 */
export function trainLogisticL1(
  samples: LRPoint[],
  lambda: number,
  lr: number,
  maxSteps: number,
  w0?: number[],
  b0?: number,
): { w: number[]; b: number; loss: number } {
  const n = samples.length
  const p = samples[0].x.length
  let w: number[] = w0 ? w0.slice() : new Array(p).fill(0)
  let b: number = b0 ?? 0

  for (let step = 0; step < maxSteps; step++) {
    const gw: number[] = new Array(p).fill(0)
    let gb = 0
    for (const s of samples) {
      const prob = logisticPredictProb(w, b, s.x)
      const err = prob - s.y
      for (let j = 0; j < p; j++) gw[j] += err * s.x[j]
      gb += err
    }
    const wNext = new Array(p)
    for (let j = 0; j < p; j++) {
      // Proximal step: gradient step on the smooth part, then prox of the
      // L1 term (soft-thresholding). The prox never overshoots, so unlike
      // L2 this stays stable at any λ — but guard non-finite anyway.
      wNext[j] = softThreshold(w[j] - lr * (gw[j] / n), lr * lambda)
    }
    const bNext = b - lr * (gb / n)
    if (!Number.isFinite(bNext) || wNext.some((v) => !Number.isFinite(v))) {
      break
    }
    w = wNext
    b = bNext
  }
  return { w, b, loss: logisticLoss(samples, w, b) }
}

/** Standardise each feature to zero mean and unit variance (per column). */
export function standardiseFeatures(samples: LRPoint[]): LRPoint[] {
  const n = samples.length
  const p = samples[0].x.length
  const mean = new Array(p).fill(0)
  for (const s of samples) for (let j = 0; j < p; j++) mean[j] += s.x[j]
  for (let j = 0; j < p; j++) mean[j] /= n
  const std = new Array(p).fill(0)
  for (const s of samples)
    for (let j = 0; j < p; j++) std[j] += (s.x[j] - mean[j]) ** 2
  for (let j = 0; j < p; j++) std[j] = Math.sqrt(std[j] / n) || 1
  return samples.map((s) => ({
    y: s.y,
    x: s.x.map((v, j) => (v - mean[j]) / std[j]) as number[],
  }))
}

/**
 * Compute the L1 or L2 regularisation path: train at a series of λ values
 * (log-spaced), warm-starting from the previous solution. Returns an
 * array of (λ, coefficients) entries with λ ascending.
 *
 * Two robustness measures, standard for ridge/lasso paths:
 *  - Features are standardised so a single λ penalises every coefficient
 *    on the same scale, and the solver stays well-conditioned.
 *  - The path is traced from large λ down to small λ, warm-starting each
 *    solve from the previous (more-regularised, smaller-weight) solution.
 *    Starting from the heavily-shrunk end keeps the warm starts tame; the
 *    result is reversed to ascending λ before returning.
 */
export function regularisationPath(
  samples: LRPoint[],
  kind: 'L1' | 'L2',
  lambdas: number[],
  opts: { lr: number; stepsPerLambda: number },
): { lambda: number; w: number[]; b: number }[] {
  const std = standardiseFeatures(samples)
  const descending = lambdas.slice().sort((a, b) => b - a)
  const path: { lambda: number; w: number[]; b: number }[] = []
  const train = kind === 'L1' ? trainLogisticL1 : trainLogisticL2
  let wPrev: number[] | undefined = undefined
  let bPrev = 0
  for (const lam of descending) {
    const { w, b } = train(std, lam, opts.lr, opts.stepsPerLambda, wPrev, bPrev)
    path.push({ lambda: lam, w: w.slice(), b })
    wPrev = w
    bPrev = b
  }
  return path.reverse()
}

/** Log-spaced grid of λ values from `lo` to `hi`, length `n`. */
export function logspace(lo: number, hi: number, n: number): number[] {
  const out: number[] = new Array(n)
  const logLo = Math.log10(lo)
  const logHi = Math.log10(hi)
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    out[i] = Math.pow(10, logLo + t * (logHi - logLo))
  }
  return out
}
