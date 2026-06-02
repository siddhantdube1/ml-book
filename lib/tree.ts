import { createRng, gauss } from './rng'

// ─── Classification trees ────────────────────────────────────────────

/** A labelled point with p features. y is an integer class {0, ..., K-1}. */
export type Point = { x: number[]; y: number }

export type TreeNode =
  | {
      kind: 'leaf'
      prediction: number
      counts: number[]
      n: number
      depth: number
    }
  | {
      kind: 'split'
      feature: number
      threshold: number
      gain: number
      left: TreeNode
      right: TreeNode
      counts: number[]
      n: number
      depth: number
    }

export type Criterion = 'gini' | 'entropy'

export type TreeOptions = {
  numClasses: number
  maxDepth: number
  minSamplesLeaf: number
  criterion?: Criterion
  /** Features to try per split (random subset). Default: all features. */
  maxFeatures?: number
}

// ─── Impurity ────────────────────────────────────────────────────────

/** Gini impurity: 1 − Σ p_c². Zero for a pure node, maximal when uniform. */
export function gini(counts: number[]): number {
  let n = 0
  for (const c of counts) n += c
  if (n === 0) return 0
  let s = 0
  for (const c of counts) {
    const p = c / n
    s += p * p
  }
  return 1 - s
}

/** Shannon entropy in bits: −Σ p_c log₂ p_c. */
export function entropy(counts: number[]): number {
  let n = 0
  for (const c of counts) n += c
  if (n === 0) return 0
  let h = 0
  for (const c of counts) {
    if (c === 0) continue
    const p = c / n
    h -= p * Math.log2(p)
  }
  return h
}

function impurityOf(counts: number[], criterion: Criterion): number {
  return criterion === 'entropy' ? entropy(counts) : gini(counts)
}

export function classCounts(points: Point[], numClasses: number): number[] {
  const counts = new Array(numClasses).fill(0)
  for (const p of points) counts[p.y]++
  return counts
}

function argmax(counts: number[]): number {
  let best = 0
  for (let i = 1; i < counts.length; i++) if (counts[i] > counts[best]) best = i
  return best
}

// ─── Best split ──────────────────────────────────────────────────────

export type Split = { feature: number; threshold: number; gain: number }

/**
 * Sweep every candidate threshold (midpoints of consecutive sorted values)
 * on each candidate feature, return the split of greatest impurity decrease
 * — or null if none reduces impurity while respecting minSamplesLeaf. The
 * sweep moves one point at a time from the right child to the left, updating
 * counts incrementally, so each feature costs O(n log n + nK). `features`
 * restricts the search to a subset of feature indices (default: all).
 */
export function bestSplit(
  points: Point[],
  opts: TreeOptions,
  features?: number[],
): Split | null {
  const K = opts.numClasses
  const crit = opts.criterion ?? 'gini'
  const minLeaf = opts.minSamplesLeaf
  const n = points.length
  if (n < 2 * minLeaf) return null
  const parent = classCounts(points, K)
  const parentImp = impurityOf(parent, crit)
  const feats =
    features ?? Array.from({ length: points[0].x.length }, (_, i) => i)

  let best: Split | null = null

  for (const feature of feats) {
    const sorted = points.slice().sort((a, b) => a.x[feature] - b.x[feature])
    const leftCounts = new Array(K).fill(0)
    const rightCounts = parent.slice()
    let nLeft = 0
    for (let i = 0; i < n - 1; i++) {
      const c = sorted[i].y
      leftCounts[c]++
      rightCounts[c]--
      nLeft++
      const nRight = n - nLeft
      // only a real threshold between two distinct feature values
      if (sorted[i].x[feature] === sorted[i + 1].x[feature]) continue
      if (nLeft < minLeaf || nRight < minLeaf) continue
      const wImp =
        (nLeft / n) * impurityOf(leftCounts, crit) +
        (nRight / n) * impurityOf(rightCounts, crit)
      const gain = parentImp - wImp
      if (best === null || gain > best.gain) {
        const threshold =
          (sorted[i].x[feature] + sorted[i + 1].x[feature]) / 2
        best = { feature, threshold, gain }
      }
    }
  }
  return best
}

// ─── Build (recursive, depth-limited) ────────────────────────────────

const EPS = 1e-12

/**
 * Pick a random subset of `k` of the `p` feature indices, in place via
 * partial Fisher–Yates. Used for the per-node feature subsampling that
 * decorrelates the trees of a random forest.
 */
function randomFeatureSubset(p: number, k: number, rng: () => number): number[] {
  const idx = Array.from({ length: p }, (_, i) => i)
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (p - i))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  return idx.slice(0, k)
}

export function buildTree(
  points: Point[],
  opts: TreeOptions,
  depth = 0,
  rng?: () => number,
): TreeNode {
  const K = opts.numClasses
  const counts = classCounts(points, K)
  const n = points.length
  const prediction = argmax(counts)
  const leaf = (): TreeNode => ({ kind: 'leaf', prediction, counts, n, depth })

  if (depth >= opts.maxDepth || n < 2 * opts.minSamplesLeaf || gini(counts) === 0)
    return leaf()

  // Feature subsampling: when maxFeatures is set below the full count and an
  // rng is supplied, each node considers a fresh random subset of features.
  const p = points[0].x.length
  const features =
    rng && opts.maxFeatures && opts.maxFeatures < p
      ? randomFeatureSubset(p, opts.maxFeatures, rng)
      : undefined

  const split = bestSplit(points, opts, features)
  if (!split || split.gain <= EPS) return leaf()

  const left = points.filter((p) => p.x[split.feature] < split.threshold)
  const right = points.filter((p) => p.x[split.feature] >= split.threshold)
  return {
    kind: 'split',
    feature: split.feature,
    threshold: split.threshold,
    gain: split.gain,
    left: buildTree(left, opts, depth + 1, rng),
    right: buildTree(right, opts, depth + 1, rng),
    counts,
    n,
    depth,
  }
}

// ─── Best-first growth with frame history (for playback) ─────────────

export type TreeFrame = { root: TreeNode; nSplits: number; lastPath: number[] }

type MutNode = {
  kind: 'leaf' | 'split'
  points: Point[]
  depth: number
  path: number[]
  counts: number[]
  prediction: number
  cachedSplit?: Split | null
  feature?: number
  threshold?: number
  gain?: number
  left?: MutNode
  right?: MutNode
}

function makeMutLeaf(
  points: Point[],
  depth: number,
  path: number[],
  K: number,
): MutNode {
  const counts = classCounts(points, K)
  return { kind: 'leaf', points, depth, path, counts, prediction: argmax(counts) }
}

function snapshot(node: MutNode): TreeNode {
  if (node.kind === 'leaf') {
    return {
      kind: 'leaf',
      prediction: node.prediction,
      counts: node.counts.slice(),
      n: node.points.length,
      depth: node.depth,
    }
  }
  return {
    kind: 'split',
    feature: node.feature as number,
    threshold: node.threshold as number,
    gain: node.gain as number,
    left: snapshot(node.left as MutNode),
    right: snapshot(node.right as MutNode),
    counts: node.counts.slice(),
    n: node.points.length,
    depth: node.depth,
  }
}

/**
 * Grow the tree best-first: at each step split the frontier leaf whose split
 * has the greatest gain. Returns the tree after 0, 1, 2, … splits, so a
 * playback can step through the construction one split at a time.
 */
export function buildTreeFrames(
  points: Point[],
  opts: TreeOptions,
  maxSplits: number,
): TreeFrame[] {
  const K = opts.numClasses
  const root = makeMutLeaf(points, 0, [], K)
  const frontier: MutNode[] = [root]
  const frames: TreeFrame[] = [{ root: snapshot(root), nSplits: 0, lastPath: [] }]

  for (let step = 1; step <= maxSplits; step++) {
    let chosen: MutNode | null = null
    let chosenSplit: Split | null = null
    for (const leaf of frontier) {
      if (leaf.kind !== 'leaf') continue
      if (leaf.depth >= opts.maxDepth) continue
      if (leaf.points.length < 2 * opts.minSamplesLeaf) continue
      if (gini(leaf.counts) === 0) continue
      if (leaf.cachedSplit === undefined) leaf.cachedSplit = bestSplit(leaf.points, opts)
      const s = leaf.cachedSplit
      if (!s || s.gain <= EPS) continue
      if (chosenSplit === null || s.gain > chosenSplit.gain) {
        chosen = leaf
        chosenSplit = s
      }
    }
    if (!chosen || !chosenSplit) break

    const { feature, threshold, gain } = chosenSplit
    const lp = chosen.points.filter((p) => p.x[feature] < threshold)
    const rp = chosen.points.filter((p) => p.x[feature] >= threshold)
    chosen.kind = 'split'
    chosen.feature = feature
    chosen.threshold = threshold
    chosen.gain = gain
    chosen.left = makeMutLeaf(lp, chosen.depth + 1, [...chosen.path, 0], K)
    chosen.right = makeMutLeaf(rp, chosen.depth + 1, [...chosen.path, 1], K)

    const idx = frontier.indexOf(chosen)
    frontier.splice(idx, 1, chosen.left, chosen.right)

    frames.push({ root: snapshot(root), nSplits: step, lastPath: chosen.path })
  }

  return frames
}

// ─── Use ─────────────────────────────────────────────────────────────

export function predict(node: TreeNode, x: number[]): number {
  let cur = node
  while (cur.kind === 'split') {
    cur = x[cur.feature] < cur.threshold ? cur.left : cur.right
  }
  return cur.prediction
}

export function accuracy(node: TreeNode, points: Point[]): number {
  if (points.length === 0) return 0
  let correct = 0
  for (const p of points) if (predict(node, p.x) === p.y) correct++
  return correct / points.length
}

export function countLeaves(node: TreeNode): number {
  if (node.kind === 'leaf') return 1
  return countLeaves(node.left) + countLeaves(node.right)
}

export function treeDepthOf(node: TreeNode): number {
  if (node.kind === 'leaf') return 0
  return 1 + Math.max(treeDepthOf(node.left), treeDepthOf(node.right))
}

// ─── Regression trees (1-D x → continuous y) ─────────────────────────

export type RegPoint = { x: number; y: number }

export type RegNode =
  | { kind: 'leaf'; value: number; n: number; depth: number }
  | {
      kind: 'split'
      threshold: number
      left: RegNode
      right: RegNode
      n: number
      depth: number
    }

function meanY(points: RegPoint[]): number {
  if (points.length === 0) return 0
  let s = 0
  for (const p of points) s += p.y
  return s / points.length
}

function sse(points: RegPoint[]): number {
  // sum of squared deviations from the mean
  const m = meanY(points)
  let s = 0
  for (const p of points) s += (p.y - m) * (p.y - m)
  return s
}

/**
 * Greedy regression tree splitting on variance reduction (equivalently, the
 * largest drop in total squared error). Each leaf predicts the mean target
 * of the points that reach it, so the tree is a piecewise-constant function.
 */
export function buildRegressionTree(
  points: RegPoint[],
  maxDepth: number,
  minLeaf: number,
  depth = 0,
): RegNode {
  const n = points.length
  const value = meanY(points)
  const leaf = (): RegNode => ({ kind: 'leaf', value, n, depth })
  if (depth >= maxDepth || n < 2 * minLeaf || sse(points) < 1e-9) return leaf()

  const sorted = points.slice().sort((a, b) => a.x - b.x)
  const parentSse = sse(points)
  let best: { threshold: number; reduction: number } | null = null
  for (let i = 0; i < n - 1; i++) {
    const nLeft = i + 1
    const nRight = n - nLeft
    if (sorted[i].x === sorted[i + 1].x) continue
    if (nLeft < minLeaf || nRight < minLeaf) continue
    const left = sorted.slice(0, nLeft)
    const right = sorted.slice(nLeft)
    const reduction = parentSse - sse(left) - sse(right)
    if (best === null || reduction > best.reduction) {
      best = { threshold: (sorted[i].x + sorted[i + 1].x) / 2, reduction }
    }
  }
  if (!best || best.reduction <= 1e-9) return leaf()

  const left = points.filter((p) => p.x < best!.threshold)
  const right = points.filter((p) => p.x >= best!.threshold)
  return {
    kind: 'split',
    threshold: best.threshold,
    left: buildRegressionTree(left, maxDepth, minLeaf, depth + 1),
    right: buildRegressionTree(right, maxDepth, minLeaf, depth + 1),
    n,
    depth,
  }
}

export function predictReg(node: RegNode, x: number): number {
  let cur = node
  while (cur.kind === 'split') {
    cur = x < cur.threshold ? cur.left : cur.right
  }
  return cur.value
}

export function countLeavesReg(node: RegNode): number {
  if (node.kind === 'leaf') return 1
  return countLeavesReg(node.left) + countLeavesReg(node.right)
}

/** Leaf intervals [x0, x1) with their predicted value, left→right, for
 *  drawing the step function over [xMin, xMax]. */
export function regSteps(
  node: RegNode,
  xMin: number,
  xMax: number,
): { x0: number; x1: number; value: number }[] {
  const out: { x0: number; x1: number; value: number }[] = []
  const walk = (n: RegNode, lo: number, hi: number) => {
    if (n.kind === 'leaf') {
      out.push({ x0: lo, x1: hi, value: n.value })
      return
    }
    walk(n.left, lo, n.threshold)
    walk(n.right, n.threshold, hi)
  }
  walk(node, xMin, xMax)
  return out
}

// ─── 2-D datasets (math coords centred near the origin) ──────────────

function shuffle<T>(arr: T[], rng: () => number): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

/** k Gaussian blobs arranged in a ring (k = 2, 3, or 4). */
export function makeBlobs2(n: number, seed: number, k = 3): Point[] {
  const rng = createRng(seed)
  const centres: [number, number][] =
    k === 2
      ? [
          [-1.5, 0.6],
          [1.5, -0.6],
        ]
      : k === 4
        ? [
            [-1.5, 1.2],
            [1.5, 1.2],
            [-1.5, -1.2],
            [1.5, -1.2],
          ]
        : [
            [-1.6, -0.8],
            [1.6, -0.8],
            [0, 1.5],
          ]
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const y = i % k
    const [cx, cy] = centres[y]
    out.push({ x: [gauss(rng, cx, 0.55), gauss(rng, cy, 0.55)], y })
  }
  return shuffle(out, rng)
}

/** Two-class checkerboard (XOR): class = (x>0) XOR (y>0). Not linearly
 *  separable; a tree carves it apart in a handful of splits. */
export function makeXOR(n: number, seed: number): Point[] {
  const rng = createRng(seed)
  const centres: [number, number][] = [
    [1.4, 1.4],
    [-1.4, 1.4],
    [-1.4, -1.4],
    [1.4, -1.4],
  ]
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const q = i % 4
    const [cx, cy] = centres[q]
    const y = cx > 0 !== cy > 0 ? 1 : 0
    out.push({ x: [gauss(rng, cx, 0.55), gauss(rng, cy, 0.55)], y })
  }
  return shuffle(out, rng)
}

/**
 * Two interleaved half-moons, scaled to fill the plotting range. `flip`
 * randomly flips that fraction of labels — class noise that a deep tree
 * will memorise, used to make the overfitting story visible.
 */
export function makeMoons2(
  n: number,
  seed: number,
  noise = 0.18,
  flip = 0,
): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  const half = Math.floor(n / 2)
  const S = 1.7
  for (let i = 0; i < half; i++) {
    const t = (i / Math.max(half - 1, 1)) * Math.PI
    out.push({
      x: [
        S * (Math.cos(t) - 0.5) + gauss(rng, 0, noise),
        S * (Math.sin(t) - 0.25) + gauss(rng, 0, noise),
      ],
      y: 0,
    })
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / Math.max(n - half - 1, 1)) * Math.PI
    out.push({
      x: [
        S * (1 - Math.cos(t) - 0.5) + gauss(rng, 0, noise),
        S * (-Math.sin(t) + 0.25) + gauss(rng, 0, noise),
      ],
      y: 1,
    })
  }
  if (flip > 0) for (const p of out) if (rng() < flip) p.y = 1 - p.y
  return shuffle(out, rng)
}
