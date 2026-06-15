import type { Point } from './datasets'

/**
 * Mean silhouette coefficient across all points.
 * s_i = (b_i - a_i) / max(a_i, b_i)
 *   a_i = mean distance to other points in same cluster
 *   b_i = mean distance to points in nearest other cluster
 * Range: [-1, 1]. Higher is better. Defined as 0 for clusters of size 1.
 */
export function silhouette(
  points: Point[],
  assignments: number[],
  K: number,
): number {
  if (K < 2) return 0
  const n = points.length
  if (n === 0) return 0

  const groups: Point[][] = Array.from({ length: K }, () => [])
  for (let i = 0; i < n; i++) groups[assignments[i]].push(points[i])

  let total = 0
  let counted = 0

  for (let i = 0; i < n; i++) {
    const ci = assignments[i]
    if (groups[ci].length < 2) continue

    let aSum = 0
    for (const q of groups[ci]) {
      if (q === points[i]) continue
      aSum += Math.hypot(points[i][0] - q[0], points[i][1] - q[1])
    }
    const a = aSum / (groups[ci].length - 1)

    let b = Infinity
    for (let k = 0; k < K; k++) {
      if (k === ci || groups[k].length === 0) continue
      let sum = 0
      for (const q of groups[k]) {
        sum += Math.hypot(points[i][0] - q[0], points[i][1] - q[1])
      }
      const meanK = sum / groups[k].length
      if (meanK < b) b = meanK
    }
    if (!isFinite(b)) continue

    total += (b - a) / Math.max(a, b)
    counted++
  }

  return counted === 0 ? 0 : total / counted
}
