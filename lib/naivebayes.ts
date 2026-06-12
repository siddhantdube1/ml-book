import { createRng, gauss } from './rng'
import type { Point } from './tree'

// ─── Gaussian naive Bayes ────────────────────────────────────────────

export type GaussianNB = {
  classes: number[]
  prior: number[] // P(y = c)
  mean: number[][] // mean[c][j]
  variance: number[][] // var[c][j] — the diagonal of the covariance
}

const VAR_FLOOR = 1e-4

/**
 * Fit a Gaussian naive Bayes model: per class, per feature, a 1-D Gaussian
 * (mean + variance), plus the class priors. One pass over the data.
 */
export function fitGaussianNB(data: Point[]): GaussianNB {
  const classes = Array.from(new Set(data.map((p) => p.y))).sort((a, b) => a - b)
  const dim = data[0].x.length
  const prior: number[] = []
  const mean: number[][] = []
  const variance: number[][] = []

  for (const c of classes) {
    const pts = data.filter((p) => p.y === c)
    prior.push(pts.length / data.length)
    const mu = new Array(dim).fill(0)
    for (const p of pts) for (let j = 0; j < dim; j++) mu[j] += p.x[j]
    for (let j = 0; j < dim; j++) mu[j] /= pts.length
    const vr = new Array(dim).fill(0)
    for (const p of pts) for (let j = 0; j < dim; j++) vr[j] += (p.x[j] - mu[j]) ** 2
    for (let j = 0; j < dim; j++) vr[j] = Math.max(vr[j] / pts.length, VAR_FLOOR)
    mean.push(mu)
    variance.push(vr)
  }
  return { classes, prior, mean, variance }
}

function logGaussian(x: number, mu: number, varr: number): number {
  return -0.5 * Math.log(2 * Math.PI * varr) - (x - mu) ** 2 / (2 * varr)
}

/** Log-posterior (up to the constant log P(x)) for each class. */
export function gnbLogPosterior(m: GaussianNB, x: number[]): number[] {
  return m.classes.map((_, ci) => {
    let s = Math.log(m.prior[ci])
    for (let j = 0; j < x.length; j++) s += logGaussian(x[j], m.mean[ci][j], m.variance[ci][j])
    return s
  })
}

/** Normalised posterior P(y = c | x), summing to 1. */
export function gnbPosterior(m: GaussianNB, x: number[]): number[] {
  const logp = gnbLogPosterior(m, x)
  const mx = Math.max(...logp)
  const ex = logp.map((v) => Math.exp(v - mx))
  const z = ex.reduce((a, b) => a + b, 0)
  return ex.map((v) => v / z)
}

export function gnbPredict(m: GaussianNB, x: number[]): number {
  const logp = gnbLogPosterior(m, x)
  let best = 0
  for (let i = 1; i < logp.length; i++) if (logp[i] > logp[best]) best = i
  return m.classes[best]
}

// ─── Datasets ────────────────────────────────────────────────────────

/**
 * Two Gaussian classes sharing an elongated covariance, offset along x. The
 * `rho` knob in [0, 1] tilts the shared cloud (its long axis rotates with
 * rho); at rho = 0 the clouds are axis-aligned, so naive Bayes' independence
 * assumption is exactly right, and as rho grows the tilt is correlation that
 * NB's upright ellipse cannot represent.
 */
export function makeCorrelatedBlobs(n: number, seed: number, rho: number): Point[] {
  const rng = createRng(seed)
  const theta = rho * Math.PI * 0.45
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const sLong = 1.45
  const sShort = 0.42
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const y = i % 2
    const cx = y === 0 ? -1.0 : 1.0
    const a = gauss(rng, 0, sLong) // along the long axis
    const b = gauss(rng, 0, sShort) // along the short axis
    out.push({ x: [cx + a * c - b * s, a * s + b * c], y })
  }
  return shuffle(out, rng)
}

/**
 * Two Gaussian classes with very different spreads (class 0 tight, class 1
 * broad), offset along x. The Bayes-optimal boundary is a closed curve, which
 * Gaussian NB's quadratic boundary can follow but a linear model (logistic
 * regression) cannot — the generative/discriminative contrast of §9.
 */
export function makeHeteroBlobs(n: number, seed: number): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const y = i % 2
    const cx = y === 0 ? -1.2 : 1.2
    const sd = y === 0 ? 0.7 : 1.7
    out.push({ x: [gauss(rng, cx, sd), gauss(rng, 0, sd)], y })
  }
  return shuffle(out, rng)
}

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// ─── Multinomial naive Bayes: the spam corpus ────────────────────────

/**
 * A small hand-authored corpus for the spam widget and problems. Counts are
 * training-set word occurrences per class; the priors are document counts.
 * Some words are spam-only or ham-only (a zero count) to make the
 * zero-frequency problem — and Laplace smoothing — concrete.
 */
export const SPAM_VOCAB = [
  'free',
  'winner',
  'money',
  'offer',
  'click',
  'prize',
  'urgent',
  'meeting',
  'project',
  'report',
  'lunch',
  'team',
] as const

// counts aligned to SPAM_VOCAB. 'winner' and 'prize' never appear in ham
// (count 0): with no smoothing, either one vetoes a message to certain spam.
// Every ham-indicative word keeps a small spam count, so a genuine ham
// message has no zeros of its own — the veto is clean to demonstrate.
export const SPAM_COUNTS = {
  spam: [18, 12, 15, 14, 16, 9, 11, 2, 2, 1, 1, 2],
  ham: [2, 0, 4, 1, 3, 0, 2, 17, 15, 13, 12, 14],
}

export const SPAM_DOC_PRIOR = { spam: 40, ham: 60 }

/**
 * Per-word log-likelihood ratio log P(w | spam) / P(w | ham) under Laplace
 * smoothing with parameter `alpha`. With alpha = 0 a zero count sends the
 * ratio to ±Infinity — the zero-frequency veto.
 */
export function spamWordLogOdds(alpha: number): number[] {
  const V = SPAM_VOCAB.length
  const spamTotal = SPAM_COUNTS.spam.reduce((a, b) => a + b, 0)
  const hamTotal = SPAM_COUNTS.ham.reduce((a, b) => a + b, 0)
  return SPAM_VOCAB.map((_, i) => {
    const pSpam = (SPAM_COUNTS.spam[i] + alpha) / (spamTotal + alpha * V)
    const pHam = (SPAM_COUNTS.ham[i] + alpha) / (hamTotal + alpha * V)
    return Math.log(pSpam) - Math.log(pHam)
  })
}

/** Log prior odds log P(spam) / P(ham). */
export function spamPriorLogOdds(): number {
  return Math.log(SPAM_DOC_PRIOR.spam) - Math.log(SPAM_DOC_PRIOR.ham)
}

/**
 * Total log-odds for a message (presence/absence per vocab word) at a given
 * smoothing level, plus the per-word contributions for the running tally.
 */
export function spamLogOdds(
  present: boolean[],
  alpha: number,
): { perWord: number[]; prior: number; total: number; pSpam: number } {
  const ratios = spamWordLogOdds(alpha)
  const perWord = present.map((on, i) => (on ? ratios[i] : 0))
  const prior = spamPriorLogOdds()
  const total = perWord.reduce((a, b) => a + b, prior)
  const pSpam = 1 / (1 + Math.exp(-total))
  return { perWord, prior, total, pSpam }
}
