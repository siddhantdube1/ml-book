import { createRng } from './rng'
import { buildTree, predict, type Point, type TreeNode } from './tree'

/**
 * A bootstrap sample of size n: indices drawn with replacement, the per-index
 * draw counts, and the out-of-bag indices (those never drawn). On average a
 * fraction (1 − 1/n)^n → 1/e ≈ 0.368 of the points land out-of-bag.
 */
export type Bootstrap = { idx: number[]; counts: number[]; oob: number[] }

export function bootstrap(n: number, seed: number): Bootstrap {
  const rng = createRng(seed)
  const idx = new Array(n)
  const counts = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    const j = Math.floor(rng() * n)
    idx[i] = j
    counts[j]++
  }
  const oob: number[] = []
  for (let i = 0; i < n; i++) if (counts[i] === 0) oob.push(i)
  return { idx, counts, oob }
}

export type Forest = {
  trees: TreeNode[]
  oob: number[][] // oob[b] = indices of the original points NOT in tree b's bag
  numClasses: number
}

export type ForestOptions = {
  numTrees: number
  numClasses: number
  maxDepth: number
  minSamplesLeaf: number
  /** Features tried per split (√p in general; 1 for the 2-D widgets). */
  maxFeatures?: number
  seed: number
}

/**
 * Train a random forest: each tree on its own bootstrap sample, with a fresh
 * random feature subset considered at every split. Records each tree's
 * out-of-bag indices so the forest can grade itself.
 */
export function trainForest(points: Point[], opts: ForestOptions): Forest {
  const trees: TreeNode[] = []
  const oob: number[][] = []
  const treeOpts = {
    numClasses: opts.numClasses,
    maxDepth: opts.maxDepth,
    minSamplesLeaf: opts.minSamplesLeaf,
    maxFeatures: opts.maxFeatures,
  }
  const n = points.length
  for (let b = 0; b < opts.numTrees; b++) {
    const bs = bootstrap(n, opts.seed + b * 1009 + 1)
    const sample = bs.idx.map((i) => points[i])
    const splitRng = createRng(opts.seed + b * 1009 + 7)
    trees.push(buildTree(sample, treeOpts, 0, splitRng))
    oob.push(bs.oob)
  }
  return { trees, oob, numClasses: opts.numClasses }
}

function argmax(a: number[]): number {
  let best = 0
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i
  return best
}

/** Vote fractions across the first `nTrees` trees (default: all). */
export function forestVoteProbs(
  forest: Forest,
  x: number[],
  nTrees?: number,
): number[] {
  const B = Math.min(nTrees ?? forest.trees.length, forest.trees.length)
  const votes = new Array(forest.numClasses).fill(0)
  for (let b = 0; b < B; b++) votes[predict(forest.trees[b], x)]++
  return votes.map((v) => v / Math.max(B, 1))
}

export function forestPredict(
  forest: Forest,
  x: number[],
  nTrees?: number,
): number {
  return argmax(forestVoteProbs(forest, x, nTrees))
}

export function forestAccuracy(
  forest: Forest,
  points: Point[],
  nTrees?: number,
): number {
  if (points.length === 0) return 0
  let correct = 0
  for (const p of points) if (forestPredict(forest, p.x, nTrees) === p.y) correct++
  return correct / points.length
}

/**
 * Out-of-bag accuracy using the first `nTrees` trees: each point is predicted
 * only by the trees whose bootstrap excluded it — an honest held-out estimate
 * with no separate validation split.
 */
export function oobAccuracy(
  forest: Forest,
  points: Point[],
  nTrees?: number,
): number {
  const B = Math.min(nTrees ?? forest.trees.length, forest.trees.length)
  const K = forest.numClasses
  const votes: number[][] = points.map(() => new Array(K).fill(0))
  const counted = new Array(points.length).fill(false)
  for (let b = 0; b < B; b++) {
    for (const i of forest.oob[b]) {
      votes[i][predict(forest.trees[b], points[i].x)]++
      counted[i] = true
    }
  }
  let correct = 0
  let total = 0
  for (let i = 0; i < points.length; i++) {
    if (!counted[i]) continue
    total++
    if (argmax(votes[i]) === points[i].y) correct++
  }
  return total === 0 ? 0 : correct / total
}

/**
 * Per-feature importance: the total impurity decrease (node size × gain) at
 * the splits using each feature, summed across the forest and normalised to
 * sum to one. Length = numFeatures.
 */
export function featureImportances(
  forest: Forest,
  numFeatures: number,
  nTrees?: number,
): number[] {
  const B = Math.min(nTrees ?? forest.trees.length, forest.trees.length)
  const imp = new Array(numFeatures).fill(0)
  const walk = (node: TreeNode) => {
    if (node.kind === 'split') {
      imp[node.feature] += node.n * node.gain
      walk(node.left)
      walk(node.right)
    }
  }
  for (let b = 0; b < B; b++) walk(forest.trees[b])
  const total = imp.reduce((a, b) => a + b, 0) || 1
  return imp.map((v) => v / total)
}
