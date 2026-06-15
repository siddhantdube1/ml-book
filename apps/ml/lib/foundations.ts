import { createRng, gauss } from './rng'

export type Pt = { x: number; y: number }

/** Noisy points scattered around the line y = slope·x + intercept. */
export function make1DLinear(
  n: number,
  seed: number,
  slope: number,
  intercept: number,
  noise: number,
  x0 = 0,
  x1 = 10,
): Pt[] {
  const rng = createRng(seed)
  const out: Pt[] = []
  for (let i = 0; i < n; i++) {
    const x = x0 + ((x1 - x0) * (i + 0.5)) / n + gauss(rng, 0, (x1 - x0) / n / 3)
    out.push({ x, y: slope * x + intercept + gauss(rng, 0, noise) })
  }
  return out
}

/** The smooth true rule behind the overfitting widgets, on x ∈ [-1, 1]. */
export const curveTrue = (x: number): number => Math.sin(2.2 * x)

/** Noisy 1-D samples of `curveTrue` on x ∈ [-1, 1] — for the workflow widgets. */
export function makeCurveData(n: number, seed: number, noise = 0.22): Pt[] {
  const rng = createRng(seed)
  const out: Pt[] = []
  for (let i = 0; i < n; i++) {
    const x = -1 + (2 * (i + 0.5)) / n
    out.push({ x, y: curveTrue(x) + gauss(rng, 0, noise) })
  }
  return out
}

/** Closed-form least-squares line: slope = cov(x,y)/var(x), intercept = ȳ − slope·x̄. */
export function leastSquaresLine(xs: number[], ys: number[]): { slope: number; intercept: number } {
  const n = xs.length
  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my)
    sxx += (xs[i] - mx) ** 2
  }
  const slope = sxx === 0 ? 0 : sxy / sxx
  return { slope, intercept: my - slope * mx }
}

/** Mean squared error of a line against points. */
export function lineMSE(pts: Pt[], slope: number, intercept: number): number {
  let s = 0
  for (const p of pts) s += (p.y - (slope * p.x + intercept)) ** 2
  return s / pts.length
}

/** Coefficient of determination R² for a line against points. */
export function rSquared(pts: Pt[], slope: number, intercept: number): number {
  const my = pts.reduce((a, p) => a + p.y, 0) / pts.length
  let ssRes = 0
  let ssTot = 0
  for (const p of pts) {
    ssRes += (p.y - (slope * p.x + intercept)) ** 2
    ssTot += (p.y - my) ** 2
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot
}
