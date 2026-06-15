import type { Point } from './datasets'
import { createRng } from './rng'

export type InitMethod = 'random' | 'kpp'

export type KMeansFrame = {
  centroids: Point[]
  assignments: number[]
  wcss: number | null
  label: string
}

function dist2(a: Point, b: Point): number {
  const dx = a[0] - b[0]
  const dy = a[1] - b[1]
  return dx * dx + dy * dy
}

function initRandom(
  points: Point[],
  k: number,
  rng: () => number,
): Point[] {
  const idxs = new Set<number>()
  while (idxs.size < k) idxs.add(Math.floor(rng() * points.length))
  return Array.from(idxs).map((i) => [points[i][0], points[i][1]] as Point)
}

function initKPP(points: Point[], k: number, rng: () => number): Point[] {
  const first = Math.floor(rng() * points.length)
  const centroids: Point[] = [[points[first][0], points[first][1]]]
  while (centroids.length < k) {
    const dists = points.map((p) => {
      let m = Infinity
      for (const c of centroids) {
        const d = dist2(p, c)
        if (d < m) m = d
      }
      return m
    })
    const sum = dists.reduce((a, b) => a + b, 0)
    let r = rng() * sum
    let acc = 0
    let picked = 0
    for (let i = 0; i < dists.length; i++) {
      acc += dists[i]
      if (acc >= r) {
        picked = i
        break
      }
    }
    centroids.push([points[picked][0], points[picked][1]])
  }
  return centroids
}

/**
 * Runs k-means and returns the full sequence of frames so callers
 * can step through, scrub, or animate the algorithm.
 */
export function runKMeans(
  points: Point[],
  k: number,
  init: InitMethod,
  seed: number,
  maxIter = 25,
): KMeansFrame[] {
  const rng = createRng(seed)
  let centroids =
    init === 'kpp' ? initKPP(points, k, rng) : initRandom(points, k, rng)

  const history: KMeansFrame[] = [
    {
      centroids: centroids.map((c) => [c[0], c[1]] as Point),
      assignments: new Array(points.length).fill(-1),
      wcss: null,
      label: 'initial centroids placed',
    },
  ]

  for (let iter = 0; iter < maxIter; iter++) {
    const assignments = points.map((p) => {
      let minD = Infinity
      let minK = 0
      for (let kk = 0; kk < k; kk++) {
        const d = dist2(p, centroids[kk])
        if (d < minD) {
          minD = d
          minK = kk
        }
      }
      return minK
    })

    let wcss = 0
    for (let i = 0; i < points.length; i++) {
      wcss += dist2(points[i], centroids[assignments[i]])
    }

    history.push({
      centroids: centroids.map((c) => [c[0], c[1]] as Point),
      assignments: assignments.slice(),
      wcss,
      label: 'assigned points to nearest centroid',
    })

    const next: Point[] = []
    for (let kk = 0; kk < k; kk++) {
      let sx = 0
      let sy = 0
      let n = 0
      for (let i = 0; i < points.length; i++) {
        if (assignments[i] === kk) {
          sx += points[i][0]
          sy += points[i][1]
          n++
        }
      }
      next.push(n === 0 ? [centroids[kk][0], centroids[kk][1]] : [sx / n, sy / n])
    }

    const moved = next.some((c, i) => dist2(c, centroids[i]) > 0.05)
    centroids = next

    history.push({
      centroids: centroids.map((c) => [c[0], c[1]] as Point),
      assignments: assignments.slice(),
      wcss,
      label: moved ? 'updated centroids to cluster means' : 'converged',
    })

    if (!moved) break
  }

  return history
}
