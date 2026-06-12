import { createRng, gauss } from './rng'
import type { Point } from './tree'

// ─── Datasets ────────────────────────────────────────────────────────

/**
 * Two gaussian blobs, class 0 left and class 1 right. `gap` sets the
 * horizontal separation between the centres and `spread` the cloud width —
 * a wide gap is cleanly separable (Widget 1), a narrow one overlaps (Widget
 * 2).
 */
export function makeSvmBlobs(
  n: number,
  seed: number,
  gap = 2.6,
  spread = 0.5,
): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const y = i % 2
    const cx = y === 0 ? -gap / 2 : gap / 2
    const cy = y === 0 ? 0.35 : -0.35
    out.push({ x: [gauss(rng, cx, spread), gauss(rng, cy, spread)], y })
  }
  return shuffle(out, rng)
}

/**
 * Concentric rings: an inner disc/ring (class 0) wrapped by an outer ring
 * (class 1). No straight line separates them, but the lift
 * (x1, x2, x1^2 + x2^2) does — the showpiece dataset for the kernel trick.
 */
export function makeCircles2(n: number, seed: number, noise = 0.14): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  const half = Math.floor(n / 2)
  for (let i = 0; i < n; i++) {
    const inner = i < half
    const r = (inner ? 0.85 : 2.1) + gauss(rng, 0, noise)
    const t = 2 * Math.PI * rng()
    out.push({
      x: [r * Math.cos(t), r * Math.sin(t)],
      y: inner ? 0 : 1,
    })
  }
  return shuffle(out, rng)
}

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// ─── Kernels (labels handled as ±1 internally) ───────────────────────

export type Kernel =
  | { kind: 'linear' }
  | { kind: 'rbf'; gamma: number }
  | { kind: 'poly'; degree: number; coef0: number }

function dot(a: number[], b: number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export function kernel(k: Kernel, a: number[], b: number[]): number {
  if (k.kind === 'linear') return dot(a, b)
  if (k.kind === 'rbf') {
    let s = 0
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i]
      s += d * d
    }
    return Math.exp(-k.gamma * s)
  }
  return Math.pow(dot(a, b) + k.coef0, k.degree)
}

// ─── Kernel soft-margin SVM via simplified SMO (the dual) ────────────

export type SVMOptions = {
  C: number
  tol?: number
  maxPasses?: number
  seed?: number
}

export type KernelSVM = {
  kern: Kernel
  b: number
  /** Support vectors only (alpha > eps) — used for prediction. */
  svX: number[][]
  svY: number[] // ±1
  svA: number[] // alpha
  /** Alpha aligned to the original training array, for circling SVs. */
  alpha: number[]
}

const ALPHA_EPS = 1e-6

/**
 * Train a soft-margin SVM with Platt's simplified SMO. Returns the dual
 * solution; the support vectors are exactly the points with alpha > 0, which
 * is why the widgets can circle them precisely.
 */
export function trainKernelSVM(
  data: Point[],
  kern: Kernel,
  opts: SVMOptions,
): KernelSVM {
  const n = data.length
  const C = opts.C
  const tol = opts.tol ?? 1e-4
  const maxPasses = opts.maxPasses ?? 30
  const rng = createRng(opts.seed ?? 1)

  const X = data.map((p) => p.x)
  const Y = data.map((p) => (p.y === 1 ? 1 : -1))
  const alpha = new Array(n).fill(0)
  let b = 0

  // Precompute the kernel (gram) matrix.
  const K: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = kernel(kern, X[i], X[j])
      K[i][j] = v
      K[j][i] = v
    }
  }

  const decision = (i: number): number => {
    let s = b
    for (let k = 0; k < n; k++) if (alpha[k] !== 0) s += alpha[k] * Y[k] * K[k][i]
    return s
  }

  let passes = 0
  while (passes < maxPasses) {
    let changed = 0
    for (let i = 0; i < n; i++) {
      const Ei = decision(i) - Y[i]
      if (
        (Y[i] * Ei < -tol && alpha[i] < C) ||
        (Y[i] * Ei > tol && alpha[i] > 0)
      ) {
        // Pick j != i at random.
        let j = Math.floor(rng() * (n - 1))
        if (j >= i) j++
        const Ej = decision(j) - Y[j]

        const ai = alpha[i]
        const aj = alpha[j]

        let L: number
        let H: number
        if (Y[i] !== Y[j]) {
          L = Math.max(0, aj - ai)
          H = Math.min(C, C + aj - ai)
        } else {
          L = Math.max(0, ai + aj - C)
          H = Math.min(C, ai + aj)
        }
        if (L === H) continue

        const eta = 2 * K[i][j] - K[i][i] - K[j][j]
        if (eta >= 0) continue

        let ajNew = aj - (Y[j] * (Ei - Ej)) / eta
        if (ajNew > H) ajNew = H
        else if (ajNew < L) ajNew = L
        if (Math.abs(ajNew - aj) < 1e-5) continue

        const aiNew = ai + Y[i] * Y[j] * (aj - ajNew)

        const b1 =
          b - Ei - Y[i] * (aiNew - ai) * K[i][i] - Y[j] * (ajNew - aj) * K[i][j]
        const b2 =
          b - Ej - Y[i] * (aiNew - ai) * K[i][j] - Y[j] * (ajNew - aj) * K[j][j]

        alpha[i] = aiNew
        alpha[j] = ajNew

        if (aiNew > 0 && aiNew < C) b = b1
        else if (ajNew > 0 && ajNew < C) b = b2
        else b = (b1 + b2) / 2

        changed++
      }
    }
    if (changed === 0) passes++
    else passes = 0
  }

  const svX: number[][] = []
  const svY: number[] = []
  const svA: number[] = []
  for (let i = 0; i < n; i++) {
    if (alpha[i] > ALPHA_EPS) {
      svX.push(X[i])
      svY.push(Y[i])
      svA.push(alpha[i])
    }
  }

  return { kern, b, svX, svY, svA, alpha }
}

/** Decision value f(x) = Σ alpha_i y_i K(sv_i, x) + b. Sign is the class. */
export function kernelDecision(m: KernelSVM, x: number[]): number {
  let s = m.b
  for (let i = 0; i < m.svX.length; i++) {
    s += m.svA[i] * m.svY[i] * kernel(m.kern, m.svX[i], x)
  }
  return s
}

export function kernelPredict(m: KernelSVM, x: number[]): 0 | 1 {
  return kernelDecision(m, x) >= 0 ? 1 : 0
}

/**
 * For a linear-kernel model, recover the primal weights w = Σ alpha_i y_i x_i
 * and bias b. The geometric margin is then 1 / ||w||.
 */
export function linearWeights(m: KernelSVM): { w: number[]; b: number } {
  const dim = m.svX.length > 0 ? m.svX[0].length : 2
  const w = new Array(dim).fill(0)
  for (let i = 0; i < m.svX.length; i++) {
    for (let d = 0; d < dim; d++) w[d] += m.svA[i] * m.svY[i] * m.svX[i][d]
  }
  return { w, b: m.b }
}

/** Geometric margin (half-width of the street) for a linear model. */
export function marginWidth(w: number[]): number {
  return 1 / Math.sqrt(dot(w, w))
}

// ─── Linear SVM via Pegasos (the §11 teaching implementation) ────────

export type LinearSVM = { w: number[]; b: number }

/**
 * Soft-margin linear SVM by stochastic sub-gradient descent (Pegasos). The
 * objective is (lambda/2)||w||^2 + (1/n)Σ max(0, 1 - y_i(w·x_i + b)); we set
 * lambda = 1 / (C·n) so larger C means weaker regularisation, matching the
 * SMO parameterisation. Bias is updated by its sub-gradient (not penalised).
 */
export function trainPegasos(
  data: Point[],
  C: number,
  epochs: number,
  seed: number,
): LinearSVM {
  const n = data.length
  const dim = data[0].x.length
  const lambda = 1 / (C * n)
  const rng = createRng(seed)
  const w = new Array(dim).fill(0)
  let b = 0
  let t = 0
  const idx = data.map((_, i) => i)
  for (let e = 0; e < epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
    }
    for (const i of idx) {
      t++
      const eta = 1 / (lambda * t)
      const p = data[i]
      const y = p.y === 1 ? 1 : -1
      const margin = y * (dot(w, p.x) + b)
      const scale = 1 - eta * lambda
      if (margin < 1) {
        for (let d = 0; d < dim; d++) w[d] = scale * w[d] + eta * y * p.x[d]
        b += eta * y
      } else {
        for (let d = 0; d < dim; d++) w[d] = scale * w[d]
      }
    }
  }
  return { w, b }
}

export function linearDecision(m: LinearSVM, x: number[]): number {
  return dot(m.w, x) + m.b
}
