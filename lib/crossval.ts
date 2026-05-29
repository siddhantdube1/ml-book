import { createRng } from './rng'

/**
 * A single split of the indices 0..n-1 into a training set and a held-out
 * test set. Every cross-validation routine in this chapter produces these.
 */
export type Fold = { trainIdx: number[]; testIdx: number[] }

/** Seeded Fisher–Yates shuffle of the indices 0..n-1. */
function shuffledIndices(n: number, seed: number): number[] {
  const rng = createRng(seed)
  const idx = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  return idx
}

/**
 * Single random train/test split with a fraction `testFrac` held out. The
 * estimate it produces is unbiased but high-variance — it depends on which
 * points happened to land in the test set.
 */
export function trainTestSplit(
  n: number,
  testFrac: number,
  seed: number,
): Fold {
  const idx = shuffledIndices(n, seed)
  const nTest = Math.min(n - 1, Math.max(1, Math.round(n * testFrac)))
  return { testIdx: idx.slice(0, nTest), trainIdx: idx.slice(nTest) }
}

/**
 * k-fold partition. Shuffle once, then cut the shuffled order into k
 * contiguous folds (the first `n % k` folds get one extra element). Each
 * fold is the test set exactly once while the other k-1 train, so every
 * index is tested exactly once and trains in k-1 folds.
 */
export function kFoldSplit(n: number, k: number, seed: number): Fold[] {
  const idx = shuffledIndices(n, seed)
  const folds: Fold[] = []
  const base = Math.floor(n / k)
  const rem = n % k
  let start = 0
  for (let f = 0; f < k; f++) {
    const size = base + (f < rem ? 1 : 0)
    const testSlice = idx.slice(start, start + size)
    const testSet = new Set(testSlice)
    const trainIdx = idx.filter((i) => !testSet.has(i))
    folds.push({ trainIdx, testIdx: testSlice })
    start += size
  }
  return folds
}

/**
 * Generic cross-validation harness. `fit` builds a model from the training
 * indices; `score` evaluates it on the held-out test indices. Returns the
 * per-fold scores and their mean and standard deviation. The model-specific
 * glue — which trainer, which metric — lives in the caller; only the
 * protocol lives here, so the same harness drives both the polynomial-ridge
 * and logistic-regression widgets.
 */
export function crossValidate<M>(
  folds: Fold[],
  fit: (trainIdx: number[]) => M,
  score: (model: M, testIdx: number[]) => number,
): { foldScores: number[]; mean: number; std: number } {
  const foldScores = folds.map((f) => score(fit(f.trainIdx), f.testIdx))
  const mean = foldScores.reduce((a, b) => a + b, 0) / foldScores.length
  const variance =
    foldScores.reduce((a, b) => a + (b - mean) * (b - mean), 0) /
    foldScores.length
  return { foldScores, mean, std: Math.sqrt(variance) }
}
