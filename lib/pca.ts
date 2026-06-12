import { createRng, gauss } from './rng'

// ─── Symmetric eigendecomposition (cyclic Jacobi) ────────────────────

/**
 * Eigendecomposition of a symmetric matrix via the cyclic Jacobi algorithm.
 * Returns eigenvalues and eigenvectors sorted by eigenvalue, descending.
 * `vectors[i]` is the unit eigenvector for `values[i]`.
 */
export function symmetricEig(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length
  const a = A.map((r) => r.slice())
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j]
    if (off < 1e-20) break

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-18) continue
        const phi = 0.5 * Math.atan2(2 * a[p][q], a[p][p] - a[q][q])
        const c = Math.cos(phi)
        const s = Math.sin(phi)
        // A <- J^T A J  (rotate columns p,q then rows p,q)
        for (let k = 0; k < n; k++) {
          const akp = a[k][p]
          const akq = a[k][q]
          a[k][p] = c * akp + s * akq
          a[k][q] = -s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k]
          const aqk = a[q][k]
          a[p][k] = c * apk + s * aqk
          a[q][k] = -s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p]
          const vkq = V[k][q]
          V[k][p] = c * vkp + s * vkq
          V[k][q] = -s * vkp + c * vkq
        }
      }
    }
  }

  const values = a.map((_, i) => a[i][i])
  const vectors = Array.from({ length: n }, (_, i) => V.map((row) => row[i]))
  const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i])
  return { values: order.map((i) => values[i]), vectors: order.map((i) => vectors[i]) }
}

// ─── PCA ─────────────────────────────────────────────────────────────

export type PCA = {
  mean: number[]
  std: number[]
  components: number[][] // rows = principal axes, descending variance
  eigenvalues: number[]
  ratios: number[]
}

export function pca(X: number[][], standardise = false): PCA {
  const n = X.length
  const d = X[0].length
  const mean = new Array(d).fill(0)
  for (const r of X) for (let j = 0; j < d; j++) mean[j] += r[j]
  for (let j = 0; j < d; j++) mean[j] /= n

  const std = new Array(d).fill(1)
  if (standardise) {
    const v = new Array(d).fill(0)
    for (const r of X) for (let j = 0; j < d; j++) v[j] += (r[j] - mean[j]) ** 2
    for (let j = 0; j < d; j++) std[j] = Math.sqrt(v[j] / n) || 1
  }

  const Z = X.map((r) => r.map((val, j) => (val - mean[j]) / std[j]))
  const C: number[][] = Array.from({ length: d }, () => new Array(d).fill(0))
  for (const r of Z) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += r[i] * r[j]
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] /= n

  const { values, vectors } = symmetricEig(C)
  const total = values.reduce((s, e) => s + Math.max(e, 0), 0) || 1
  return { mean, std, components: vectors, eigenvalues: values, ratios: values.map((e) => e / total) }
}

/** Project rows of X onto the top-k components → n × k scores. */
export function project(model: PCA, X: number[][], k: number): number[][] {
  return X.map((r) => {
    const z = r.map((val, j) => (val - model.mean[j]) / model.std[j])
    const out: number[] = []
    for (let c = 0; c < k; c++) {
      let s = 0
      for (let j = 0; j < z.length; j++) s += z[j] * model.components[c][j]
      out.push(s)
    }
    return out
  })
}

/** Reconstruct rows of X from their top-k projection, back in original space. */
export function reconstruct(model: PCA, X: number[][], k: number): number[][] {
  const scores = project(model, X, k)
  const d = model.mean.length
  return scores.map((sc) => {
    const z = new Array(d).fill(0)
    for (let c = 0; c < k; c++) for (let j = 0; j < d; j++) z[j] += sc[c] * model.components[c][j]
    return z.map((val, j) => val * model.std[j] + model.mean[j])
  })
}

// ─── Synthetic datasets ──────────────────────────────────────────────

/** An elongated, tilted 2-D Gaussian cloud — the long axis is PC1. */
export function correlatedCloud2D(n: number, seed: number): number[][] {
  const rng = createRng(seed)
  const ang = 0.5 // ~28.6°
  const c = Math.cos(ang)
  const s = Math.sin(ang)
  const sLong = 2.5
  const sShort = 0.72
  const out: number[][] = []
  for (let i = 0; i < n; i++) {
    const a = gauss(rng, 0, sLong)
    const b = gauss(rng, 0, sShort)
    out.push([a * c - b * s, a * s + b * c])
  }
  return out
}

/**
 * k Gaussian clusters in d dimensions. The cluster centres span a 2-plane
 * tilted across all features, so the clusters separate cleanly in the top two
 * principal components but overlap in any single pair of raw features.
 */
export function highDimClusters(
  n: number,
  d: number,
  k: number,
  seed: number,
): { X: number[][]; y: number[] } {
  const rng = createRng(seed)
  // two random orthonormal-ish directions hold the cluster centres
  const dir = (off: number) => Array.from({ length: d }, (_, j) => Math.cos(off + (j * 2.3)))
  const u = dir(0.4)
  const w = dir(1.9)
  const centres: number[][] = []
  for (let c = 0; c < k; c++) {
    const t = (c / k) * 2 * Math.PI
    const ca = Math.cos(t) * 6
    const cb = Math.sin(t) * 6
    centres.push(u.map((uj, j) => ca * uj + cb * w[j]))
  }
  const X: number[][] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const c = i % k
    X.push(centres[c].map((mj) => mj + gauss(rng, 0, 2.6)))
    y.push(c)
  }
  return { X, y }
}

/**
 * A d-dimensional Gaussian whose variance decays geometrically across a random
 * (non-axis-aligned) orthonormal basis — so PCA recovers a smooth scree with
 * the spectrum concentrated in the first few components.
 */
export function decayingGaussian(n: number, d: number, seed: number): number[][] {
  const rng = createRng(seed)
  // a random orthonormal basis = eigenvectors of a random symmetric matrix
  const S: number[][] = Array.from({ length: d }, () => new Array(d).fill(0))
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      const v = gauss(rng, 0, 1)
      S[i][j] = v
      S[j][i] = v
    }
  }
  const basis = symmetricEig(S).vectors // d orthonormal directions
  const sigma = Array.from({ length: d }, (_, j) => 4 * Math.pow(0.62, j))
  const out: number[][] = []
  for (let i = 0; i < n; i++) {
    const coeff = sigma.map((sj) => gauss(rng, 0, sj))
    const row = new Array(d).fill(0)
    for (let j = 0; j < d; j++) for (let c = 0; c < d; c++) row[c] += coeff[j] * basis[j][c]
    out.push(row)
  }
  return out
}
