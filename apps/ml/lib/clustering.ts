import type { Point } from './datasets'

// ─── Hierarchical (agglomerative) clustering ─────────────────────────

export type Linkage = 'single' | 'complete' | 'average' | 'ward'

/** One merge in the agglomeration: child node ids a, b joined at `height`. */
export type Merge = { a: number; b: number; height: number; size: number }

function sqDist(p: Point, q: Point): number {
  const dx = p[0] - q[0]
  const dy = p[1] - q[1]
  return dx * dx + dy * dy
}

/**
 * Agglomerative clustering via Lance–Williams. Leaves are node ids 0..n-1; the
 * t-th merge creates node id n+t. Returns the n-1 merges in order — a
 * dendrogram. Heights are Euclidean distances for single/complete/average and
 * the Ward merge cost (sqrt) for ward, so they are non-decreasing.
 */
export function agglomerative(points: Point[], linkage: Linkage): Merge[] {
  const n = points.length
  const ward = linkage === 'ward'
  const totalNodes = 2 * n
  const size = new Array(totalNodes).fill(0)
  for (let i = 0; i < n; i++) size[i] = 1

  const active = new Set<number>()
  for (let i = 0; i < n; i++) active.add(i)

  const dist = new Map<number, number>()
  const key = (i: number, j: number) => (i < j ? i * totalNodes + j : j * totalNodes + i)

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d2 = sqDist(points[i], points[j])
      dist.set(key(i, j), ward ? d2 : Math.sqrt(d2))
    }
  }

  const merges: Merge[] = []
  let nextId = n

  while (active.size > 1) {
    const arr = [...active]
    let best = Infinity
    let bi = -1
    let bj = -1
    for (let a = 0; a < arr.length; a++) {
      for (let b = a + 1; b < arr.length; b++) {
        const d = dist.get(key(arr[a], arr[b]))!
        if (d < best) {
          best = d
          bi = arr[a]
          bj = arr[b]
        }
      }
    }

    const ni = size[bi]
    const nj = size[bj]
    const newId = nextId++
    size[newId] = ni + nj
    merges.push({ a: bi, b: bj, height: ward ? Math.sqrt(best) : best, size: ni + nj })

    for (const k of arr) {
      if (k === bi || k === bj) continue
      const dik = dist.get(key(bi, k))!
      const djk = dist.get(key(bj, k))!
      let dnew: number
      if (linkage === 'single') dnew = Math.min(dik, djk)
      else if (linkage === 'complete') dnew = Math.max(dik, djk)
      else if (linkage === 'average') dnew = (ni * dik + nj * djk) / (ni + nj)
      else {
        const nk = size[k]
        dnew = ((ni + nk) * dik + (nj + nk) * djk - nk * best) / (ni + nj + nk)
      }
      dist.set(key(newId, k), dnew)
    }

    active.delete(bi)
    active.delete(bj)
    active.add(newId)
  }

  return merges
}

function makeUnionFind(n: number) {
  const parent = Array.from({ length: n }, (_, i) => i)
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]
      x = parent[x]
    }
    return x
  }
  return { find, union: (a: number, b: number) => (parent[find(a)] = find(b)) }
}

function labelsFromKeptMerges(merges: Merge[], n: number, keep: number): number[] {
  const uf = makeUnionFind(n)
  const rep = new Array(n + merges.length).fill(-1)
  for (let i = 0; i < n; i++) rep[i] = i
  for (let t = 0; t < merges.length; t++) {
    rep[n + t] = rep[merges[t].a]
    if (t < keep) uf.union(rep[merges[t].a], rep[merges[t].b])
  }
  const map = new Map<number, number>()
  let next = 0
  return Array.from({ length: n }, (_, i) => {
    const r = uf.find(i)
    if (!map.has(r)) map.set(r, next++)
    return map.get(r)!
  })
}

/** Cut the dendrogram into exactly k clusters → a label per point. */
export function cutTreeK(merges: Merge[], n: number, k: number): number[] {
  return labelsFromKeptMerges(merges, n, Math.max(0, n - k))
}

/** Cut at a height threshold (merge only below it) → a label per point. */
export function cutTreeHeight(merges: Merge[], n: number, height: number): number[] {
  let keep = 0
  while (keep < merges.length && merges[keep].height < height) keep++
  return labelsFromKeptMerges(merges, n, keep)
}

// ─── DBSCAN (density-based) ──────────────────────────────────────────

export type PointRole = 'core' | 'border' | 'noise'

export type DBSCANResult = {
  labels: number[] // cluster id per point; -1 = noise
  roles: PointRole[]
  numClusters: number
}

const UNVISITED = -2
const NOISE = -1

/**
 * DBSCAN (Ester et al.). A point is core if at least `minPts` points
 * (including itself) lie within `eps`; clusters grow by chaining core points
 * through one another's neighbourhoods; everything else is noise.
 */
export function dbscan(points: Point[], eps: number, minPts: number): DBSCANResult {
  const n = points.length
  const eps2 = eps * eps
  const labels = new Array(n).fill(UNVISITED)
  const isCore = new Array(n).fill(false)

  const region = (i: number): number[] => {
    const out: number[] = []
    for (let j = 0; j < n; j++) if (sqDist(points[i], points[j]) <= eps2) out.push(j)
    return out
  }

  let cid = -1
  for (let i = 0; i < n; i++) {
    if (labels[i] !== UNVISITED) continue
    const nb = region(i)
    if (nb.length < minPts) {
      labels[i] = NOISE
      continue
    }
    cid++
    isCore[i] = true
    labels[i] = cid
    const seeds = nb.filter((q) => q !== i)
    for (let s = 0; s < seeds.length; s++) {
      const q = seeds[s]
      if (labels[q] === NOISE) labels[q] = cid // border absorbed from earlier noise
      if (labels[q] !== UNVISITED) continue
      labels[q] = cid
      const qnb = region(q)
      if (qnb.length >= minPts) {
        isCore[q] = true
        for (const x of qnb) if (labels[x] === UNVISITED || labels[x] === NOISE) seeds.push(x)
      }
    }
  }

  const roles: PointRole[] = labels.map((l, i) =>
    l === NOISE ? 'noise' : isCore[i] ? 'core' : 'border',
  )
  return { labels, roles, numClusters: cid + 1 }
}

/** Each point's distance to its k-th nearest neighbour, sorted ascending. */
export function kDistances(points: Point[], k: number): number[] {
  const n = points.length
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const ds: number[] = []
    for (let j = 0; j < n; j++) if (j !== i) ds.push(Math.sqrt(sqDist(points[i], points[j])))
    ds.sort((a, b) => a - b)
    out.push(ds[Math.min(k - 1, ds.length - 1)])
  }
  return out.sort((a, b) => a - b)
}
