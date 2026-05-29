import { createRng, gauss } from './rng'
import { sigmoid } from './logistic'

/**
 * A single scored example: the model's predicted score (a probability in
 * [0, 1]) paired with the true binary label. Every metric in this module
 * is derived from an array of these — one source of truth, so a confusion
 * matrix and an ROC point computed for the same data can never disagree.
 */
export type Scored = { score: number; label: 0 | 1 }

export type Confusion = { tp: number; fp: number; fn: number; tn: number }

export type RocPoint = { fpr: number; tpr: number; threshold: number }
export type PrPoint = { recall: number; precision: number; threshold: number }

export type CalibrationBin = {
  lo: number
  hi: number
  count: number
  meanPredicted: number // mean score of samples falling in the bin
  observedFreq: number // fraction of those samples that are positive
}

// ─── Confusion + scalar metrics ──────────────────────────────────────
// Convention throughout: predict positive when score >= threshold.

export function confusionAt(data: Scored[], threshold: number): Confusion {
  let tp = 0
  let fp = 0
  let fn = 0
  let tn = 0
  for (const d of data) {
    const predictedPositive = d.score >= threshold
    if (d.label === 1) {
      if (predictedPositive) tp++
      else fn++
    } else {
      if (predictedPositive) fp++
      else tn++
    }
  }
  return { tp, fp, fn, tn }
}

export function precision(c: Confusion): number {
  const denom = c.tp + c.fp
  return denom === 0 ? 0 : c.tp / denom
}

export function recall(c: Confusion): number {
  const denom = c.tp + c.fn
  return denom === 0 ? 0 : c.tp / denom
}

export function f1(c: Confusion): number {
  const p = precision(c)
  const r = recall(c)
  return p + r === 0 ? 0 : (2 * p * r) / (p + r)
}

export function accuracy(c: Confusion): number {
  const total = c.tp + c.fp + c.fn + c.tn
  return total === 0 ? 0 : (c.tp + c.tn) / total
}

/** False-positive rate, fp / (fp + tn). The x-axis of the ROC curve. */
export function fpr(c: Confusion): number {
  const denom = c.fp + c.tn
  return denom === 0 ? 0 : c.fp / denom
}

// ─── Curves ──────────────────────────────────────────────────────────
// Sort by score descending once, then sweep the threshold from high to
// low, accumulating true/false positives. Tied scores are consumed as a
// group so the curve is correct even when several samples share a score.

function sortedDesc(data: Scored[]): Scored[] {
  return data.slice().sort((a, b) => b.score - a.score)
}

/**
 * ROC curve: false-positive rate (x) versus true-positive rate (y) as the
 * threshold sweeps from +∞ down to −∞. Spans (0,0) → (1,1).
 */
export function rocCurve(data: Scored[]): RocPoint[] {
  const P = data.reduce((s, d) => s + d.label, 0)
  const N = data.length - P
  if (P === 0 || N === 0) {
    return [
      { fpr: 0, tpr: 0, threshold: Infinity },
      { fpr: 1, tpr: 1, threshold: -Infinity },
    ]
  }
  const sorted = sortedDesc(data)
  const pts: RocPoint[] = [{ fpr: 0, tpr: 0, threshold: Infinity }]
  let tp = 0
  let fp = 0
  let i = 0
  while (i < sorted.length) {
    const thr = sorted[i].score
    while (i < sorted.length && sorted[i].score === thr) {
      if (sorted[i].label === 1) tp++
      else fp++
      i++
    }
    pts.push({ fpr: fp / N, tpr: tp / P, threshold: thr })
  }
  return pts
}

/**
 * Precision-recall curve. Recall (x) versus precision (y) as the threshold
 * sweeps. Anchored at recall 0 with precision 1 by convention.
 */
export function prCurve(data: Scored[]): PrPoint[] {
  const P = data.reduce((s, d) => s + d.label, 0)
  if (P === 0) return [{ recall: 0, precision: 1, threshold: Infinity }]
  const sorted = sortedDesc(data)
  const pts: PrPoint[] = [{ recall: 0, precision: 1, threshold: Infinity }]
  let tp = 0
  let fp = 0
  let i = 0
  while (i < sorted.length) {
    const thr = sorted[i].score
    while (i < sorted.length && sorted[i].score === thr) {
      if (sorted[i].label === 1) tp++
      else fp++
      i++
    }
    const prec = tp + fp === 0 ? 1 : tp / (tp + fp)
    pts.push({ recall: tp / P, precision: prec, threshold: thr })
  }
  return pts
}

/**
 * Trapezoidal area under the ROC curve, computed over exactly the points
 * the widget draws — so the displayed AUC and the rendered curve never
 * disagree.
 */
export function aucFromRoc(roc: RocPoint[]): number {
  let area = 0
  for (let i = 1; i < roc.length; i++) {
    area += (roc[i].fpr - roc[i - 1].fpr) * (roc[i].tpr + roc[i - 1].tpr) * 0.5
  }
  return area
}

/**
 * Rank-based (Mann-Whitney) AUC: the probability that a randomly chosen
 * positive scores above a randomly chosen negative, with ties counting a
 * half. Equals aucFromRoc up to tie handling — used as the §10 / Problem 4
 * cross-check.
 */
export function aucRank(data: Scored[]): number {
  const sorted = data.slice().sort((a, b) => a.score - b.score)
  let rankSumPos = 0
  let P = 0
  let i = 0
  // Average ranks within tied groups (ranks are 1-based).
  while (i < sorted.length) {
    let j = i
    while (j < sorted.length && sorted[j].score === sorted[i].score) j++
    const avgRank = (i + 1 + j) / 2 // mean of ranks i+1 .. j
    for (let k = i; k < j; k++) {
      if (sorted[k].label === 1) {
        rankSumPos += avgRank
        P++
      }
    }
    i = j
  }
  const N = data.length - P
  if (P === 0 || N === 0) return 0.5
  return (rankSumPos - (P * (P + 1)) / 2) / (P * N)
}

/** Average precision: area under the PR curve (sum of precision × Δrecall). */
export function averagePrecision(pr: PrPoint[]): number {
  let ap = 0
  for (let i = 1; i < pr.length; i++) {
    ap += (pr[i].recall - pr[i - 1].recall) * pr[i].precision
  }
  return ap
}

// ─── Calibration ─────────────────────────────────────────────────────

export function calibrationBins(
  data: Scored[],
  nBins: number,
): CalibrationBin[] {
  const bins: CalibrationBin[] = []
  for (let b = 0; b < nBins; b++) {
    const lo = b / nBins
    const hi = (b + 1) / nBins
    let count = 0
    let sumScore = 0
    let sumLabel = 0
    for (const d of data) {
      // Last bin is closed on the right so score === 1 lands somewhere.
      const inBin =
        b === nBins - 1
          ? d.score >= lo && d.score <= hi
          : d.score >= lo && d.score < hi
      if (inBin) {
        count++
        sumScore += d.score
        sumLabel += d.label
      }
    }
    bins.push({
      lo,
      hi,
      count,
      meanPredicted: count === 0 ? (lo + hi) / 2 : sumScore / count,
      observedFreq: count === 0 ? 0 : sumLabel / count,
    })
  }
  return bins
}

/**
 * Expected calibration error: the count-weighted mean gap between
 * predicted and observed frequency across non-empty bins. 0 is perfect
 * calibration; larger means the probabilities are less trustworthy.
 */
export function expectedCalibrationError(bins: CalibrationBin[]): number {
  let total = 0
  let weighted = 0
  for (const b of bins) {
    if (b.count === 0) continue
    total += b.count
    weighted += b.count * Math.abs(b.meanPredicted - b.observedFreq)
  }
  return total === 0 ? 0 : weighted / total
}

// ─── Score generators (single source of truth for the widgets) ────────

export type Distortion = 'none' | 'overconfident' | 'underconfident'

const CLAMP = 1e-6
function logit(p: number): number {
  const q = Math.min(1 - CLAMP, Math.max(CLAMP, p))
  return Math.log(q / (1 - q))
}

/**
 * Generate scored examples with a known, controllable structure.
 *
 * For each sample a latent "true probability" is drawn as
 *   p = sigmoid(bias + N(0, spread)),
 * and the label is drawn as Bernoulli(p). Built this way the latent p is
 * perfectly *calibrated*: among samples with true probability s, a
 * fraction s really are positive.
 *
 *  - `baseRate` sets the latent bias as logit(baseRate) — the knob for
 *    class imbalance. The *realised* fraction of positives runs somewhat
 *    higher than `baseRate` under non-zero spread (Jensen's inequality
 *    pulls the mean of sigmoid above sigmoid of the mean), so widgets
 *    report the observed positive rate computed from the data rather than
 *    the parameter.
 *  - `spread` sets how decisively the latent separates the classes, i.e.
 *    the achievable ranking quality (AUC).
 *  - `distortion` reports a *monotonic transform* of p as the score while
 *    the label still comes from the true p. Because the transform is
 *    monotonic the ranking — and therefore the AUC — is unchanged, but the
 *    reported probabilities are no longer calibrated. 'overconfident'
 *    pushes scores toward 0 and 1; 'underconfident' pulls them toward 0.5.
 */
export function makeScores(
  n: number,
  seed: number,
  opts: { baseRate?: number; spread?: number; distortion?: Distortion } = {},
): Scored[] {
  const { baseRate = 0.5, spread = 1.6, distortion = 'none' } = opts
  const rng = createRng(seed)
  const bias = logit(baseRate)
  const out: Scored[] = []
  for (let i = 0; i < n; i++) {
    const trueP = sigmoid(bias + gauss(rng, 0, spread))
    const label: 0 | 1 = rng() < trueP ? 1 : 0
    let score = trueP
    if (distortion === 'overconfident') score = sigmoid(2.2 * logit(trueP))
    else if (distortion === 'underconfident') score = sigmoid(0.45 * logit(trueP))
    out.push({ score, label })
  }
  return out
}
