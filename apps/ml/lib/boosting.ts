import {
  buildRegressionTree,
  predictReg,
  type RegNode,
  type RegPoint,
  type Point,
} from './tree'
import { sigmoid } from './logistic'

export type GBOptions = {
  numTrees: number
  learningRate: number
  maxDepth: number
  minLeaf: number
}

// ─── Regression: additive trees fit to squared-loss residuals ────────

export type GBRegressor = { init: number; lr: number; trees: RegNode[] }

/**
 * Gradient boosting for squared-error regression. Start at the mean, then
 * repeatedly fit a shallow tree to the residuals y − F (the negative gradient
 * of squared loss) and add a shrunk copy of it to the ensemble. The first m
 * trees are exactly the ensemble after m rounds, so widgets can step through
 * the sequence by passing nTrees to gbPredict.
 */
export function trainGBRegressor(
  X: number[][],
  y: number[],
  opts: GBOptions,
): GBRegressor {
  const n = y.length
  const init = y.reduce((a, b) => a + b, 0) / n
  const F = new Array(n).fill(init)
  const trees: RegNode[] = []
  for (let m = 0; m < opts.numTrees; m++) {
    const pts: RegPoint[] = X.map((xi, i) => ({ x: xi, y: y[i] - F[i] }))
    const tree = buildRegressionTree(pts, opts.maxDepth, opts.minLeaf)
    trees.push(tree)
    for (let i = 0; i < n; i++) F[i] += opts.learningRate * predictReg(tree, X[i])
  }
  return { init, lr: opts.learningRate, trees }
}

export function gbPredict(
  model: GBRegressor,
  x: number[],
  nTrees?: number,
): number {
  const B = Math.min(nTrees ?? model.trees.length, model.trees.length)
  let f = model.init
  for (let b = 0; b < B; b++) f += model.lr * predictReg(model.trees[b], x)
  return f
}

// ─── Classification: additive trees in log-odds space, logistic loss ─

export type GBClassifier = { init: number; lr: number; trees: RegNode[] }

/**
 * Gradient boosting for binary classification. Model the log-odds additively;
 * the pseudo-residual for logistic loss is the clean probability error
 * y − σ(F). Each round fits a shallow tree to that pseudo-residual and adds a
 * shrunk copy. (This is the gradient form; the leaf values are the gradients,
 * not the Newton step the production libraries use — simpler, and enough.)
 */
export function trainGBClassifier(
  points: Point[],
  opts: GBOptions,
): GBClassifier {
  const n = points.length
  const X = points.map((p) => p.x)
  const y = points.map((p) => p.y)
  const pbar = y.reduce((a, b) => a + b, 0) / n
  const clamped = Math.min(1 - 1e-6, Math.max(1e-6, pbar))
  const init = Math.log(clamped / (1 - clamped))
  const F = new Array(n).fill(init)
  const trees: RegNode[] = []
  for (let m = 0; m < opts.numTrees; m++) {
    const pts: RegPoint[] = X.map((xi, i) => ({ x: xi, y: y[i] - sigmoid(F[i]) }))
    const tree = buildRegressionTree(pts, opts.maxDepth, opts.minLeaf)
    trees.push(tree)
    for (let i = 0; i < n; i++) F[i] += opts.learningRate * predictReg(tree, X[i])
  }
  return { init, lr: opts.learningRate, trees }
}

export function gbClassProb(
  model: GBClassifier,
  x: number[],
  nTrees?: number,
): number {
  const B = Math.min(nTrees ?? model.trees.length, model.trees.length)
  let f = model.init
  for (let b = 0; b < B; b++) f += model.lr * predictReg(model.trees[b], x)
  return sigmoid(f)
}

export function gbClassPredict(
  model: GBClassifier,
  x: number[],
  nTrees?: number,
): 0 | 1 {
  return gbClassProb(model, x, nTrees) >= 0.5 ? 1 : 0
}
