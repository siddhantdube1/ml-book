import { createRng, gauss } from './rng'
import type { Point } from './tree'

// ─── Scaling ─────────────────────────────────────────────────────────

export type Scaler = { mean: number[]; std: number[] }

export function fitScaler(rows: number[][]): Scaler {
  const dim = rows[0].length
  const mean = new Array(dim).fill(0)
  const std = new Array(dim).fill(0)
  for (const r of rows) for (let j = 0; j < dim; j++) mean[j] += r[j]
  for (let j = 0; j < dim; j++) mean[j] /= rows.length
  for (const r of rows) for (let j = 0; j < dim; j++) std[j] += (r[j] - mean[j]) ** 2
  for (let j = 0; j < dim; j++) std[j] = Math.sqrt(std[j] / rows.length) || 1
  return { mean, std }
}

export function applyScaler(s: Scaler, x: number[]): number[] {
  return x.map((v, j) => (v - s.mean[j]) / s.std[j])
}

// ─── Basis expansion ─────────────────────────────────────────────────

export type PolyTerm = 'x1' | 'x2' | 'x1^2' | 'x2^2' | 'x1*x2'
export const ALL_TERMS: PolyTerm[] = ['x1', 'x2', 'x1^2', 'x2^2', 'x1*x2']

export function expand(x: [number, number], terms: PolyTerm[]): number[] {
  return terms.map((t) => {
    switch (t) {
      case 'x1':
        return x[0]
      case 'x2':
        return x[1]
      case 'x1^2':
        return x[0] * x[0]
      case 'x2^2':
        return x[1] * x[1]
      case 'x1*x2':
        return x[0] * x[1]
    }
  })
}

// ─── Cyclical encoding ───────────────────────────────────────────────

export function cyclical(t: number, period: number): [number, number] {
  const a = (2 * Math.PI * t) / period
  return [Math.sin(a), Math.cos(a)]
}

// ─── Categorical encoding ────────────────────────────────────────────

export function oneHot(value: number, k: number): number[] {
  const v = new Array(k).fill(0)
  v[value] = 1
  return v
}

/** Mean label per category (the target-encoding lookup table). */
export function targetEncodeTable(values: number[], labels: number[], k: number): number[] {
  const sum = new Array(k).fill(0)
  const cnt = new Array(k).fill(0)
  for (let i = 0; i < values.length; i++) {
    sum[values[i]] += labels[i]
    cnt[values[i]]++
  }
  const base = labels.reduce((a, b) => a + b, 0) / labels.length
  return sum.map((s, c) => (cnt[c] > 0 ? s / cnt[c] : base))
}

// ─── A small kNN classifier ──────────────────────────────────────────

function dist2(a: number[], b: number[]): number {
  let s = 0
  for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2
  return s
}

/** Indices of the k nearest training points to x (optionally in scaled space). */
export function knnNeighbours(train: Point[], x: number[], k: number, scaler?: Scaler): number[] {
  const xx = scaler ? applyScaler(scaler, x) : x
  const d = train.map((p, i) => ({ i, d: dist2(scaler ? applyScaler(scaler, p.x) : p.x, xx) }))
  d.sort((a, b) => a.d - b.d)
  return d.slice(0, k).map((o) => o.i)
}

export function knnPredict(train: Point[], x: number[], k: number, scaler?: Scaler): number {
  const nn = knnNeighbours(train, x, k, scaler)
  let votes = 0
  for (const i of nn) votes += train[i].y
  return votes * 2 >= k ? 1 : 0
}

// ─── A general logistic-regression fit (any number of features) ──────

/**
 * Logistic regression by full-batch gradient descent on standardised
 * features. Returns a predictor mapping a raw feature vector to P(y = 1).
 */
export function fitLogistic(
  X: number[][],
  y: number[],
  steps = 400,
  lr = 0.5,
): (x: number[]) => number {
  const n = X.length
  const dim = X[0].length
  const scaler = fitScaler(X)
  const Z = X.map((r) => applyScaler(scaler, r))
  const w = new Array(dim).fill(0)
  let b = 0
  const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))
  for (let s = 0; s < steps; s++) {
    const gw = new Array(dim).fill(0)
    let gb = 0
    for (let i = 0; i < n; i++) {
      let z = b
      for (let j = 0; j < dim; j++) z += w[j] * Z[i][j]
      const err = sigmoid(z) - y[i]
      for (let j = 0; j < dim; j++) gw[j] += err * Z[i][j]
      gb += err
    }
    for (let j = 0; j < dim; j++) w[j] -= (lr * gw[j]) / n
    b -= (lr * gb) / n
  }
  return (x: number[]) => {
    const z0 = applyScaler(scaler, x)
    let z = b
    for (let j = 0; j < dim; j++) z += w[j] * z0[j]
    return sigmoid(z)
  }
}

// ─── Synthetic datasets ──────────────────────────────────────────────

/**
 * Two features on wildly different scales: x1 in [0, 1000], x2 in [0, 1]. The
 * class depends on BOTH (a diagonal boundary in normalised space), so a
 * distance-based model that ignores scale — letting x1 dominate — gets it
 * badly wrong until the features are standardised.
 */
export function makeScaleData(n: number, seed: number): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 1000
    const x2 = rng()
    const score = x1 / 1000 + x2 + gauss(rng, 0, 0.06)
    out.push({ x: [x1, x2], y: score > 1 ? 1 : 0 })
  }
  return out
}

/** Hour-of-day (0–23) with a wrap-around target: busy late night / early. */
export function makeHourlyData(n: number, seed: number): { hour: number; y: number }[] {
  const rng = createRng(seed)
  const out: { hour: number; y: number }[] = []
  for (let i = 0; i < n; i++) {
    const hour = Math.floor(rng() * 24)
    // "busy" near midnight: hours 22,23,0,1,2,3
    const busy = hour >= 22 || hour <= 3
    const y = (busy ? 0.85 : 0.12) + gauss(rng, 0, 0.05) > 0.5 ? 1 : 0
    out.push({ hour, y })
  }
  return out
}

/** Hand-authored category rates for the categorical-encoding widget. */
export const CITY_RATES: { name: string; rate: number }[] = [
  { name: 'Lagos', rate: 0.71 },
  { name: 'Oslo', rate: 0.18 },
  { name: 'Lima', rate: 0.44 },
  { name: 'Cairo', rate: 0.63 },
  { name: 'Perth', rate: 0.29 },
  { name: 'Quito', rate: 0.52 },
]
