'use client'

import { useMemo, useState } from 'react'
import {
  makeSparseDataset,
  trainLogisticL2,
  logspace,
  type LRPoint,
} from '@/lib/regularisation'
import { kFoldSplit } from '@/lib/crossval'
import { sigmoid } from '@/lib/logistic'

const W = 640
const H = 380
const PAD_L = 48
const PAD_R = 20
const PAD_T = 24
const PAD_B = 46
const Y_MIN = 0.25
const Y_MAX = 1.0

const N = 60
const N_LAMBDA = 15
const LMIN = 1e-3
const LMAX = 3
const STEPS = 200

const CV_COLOR = 'var(--accent)'
const TRAIN_COLOR = 'var(--ink-muted)'

function accuracyOf(model: { w: number[]; b: number }, pts: LRPoint[]): number {
  let c = 0
  for (const s of pts) {
    let z = model.b
    for (let j = 0; j < model.w.length; j++) z += model.w[j] * s.x[j]
    if ((sigmoid(z) >= 0.5 ? 1 : 0) === s.y) c++
  }
  return c / pts.length
}

export default function CVTuning() {
  const [k, setK] = useState(5)
  const [showTrain, setShowTrain] = useState(true)

  const data = useMemo(
    () => makeSparseDataset(N, 3, 20, [1.8, -1.4, 1.0], 7, 0.5),
    [],
  )
  const lambdas = useMemo(() => logspace(LMIN, LMAX, N_LAMBDA), [])

  const { cvMean, cvStd, trainAcc, bestIdx } = useMemo(() => {
    const folds = kFoldSplit(N, k, 1)
    const desc = lambdas.slice().sort((a, b) => b - a)
    // CV: per fold, warm-start descending λ (large → small) for stable,
    // well-converged fits at small λ — the same trick the regularisation-path
    // widget uses.
    const perFold: Map<number, number[]> = new Map(lambdas.map((l) => [l, []]))
    for (const fold of folds) {
      const train = fold.trainIdx.map((i) => data[i])
      const test = fold.testIdx.map((i) => data[i])
      let wPrev: number[] | undefined
      let bPrev = 0
      for (const lam of desc) {
        const m = trainLogisticL2(train, lam, 0.4, STEPS, wPrev, bPrev)
        wPrev = m.w
        bPrev = m.b
        perFold.get(lam)!.push(accuracyOf(m, test))
      }
    }
    const cvMean: number[] = []
    const cvStd: number[] = []
    for (const lam of lambdas) {
      const xs = perFold.get(lam)!
      const m = xs.reduce((a, b) => a + b, 0) / xs.length
      const sd = Math.sqrt(
        xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length,
      )
      cvMean.push(m)
      cvStd.push(sd)
    }
    // Training accuracy: fit on all data, score on all data (warm-started).
    const trainAcc: number[] = new Array(lambdas.length)
    let wP: number[] | undefined
    let bP = 0
    for (const lam of desc) {
      const m = trainLogisticL2(data, lam, 0.4, STEPS, wP, bP)
      wP = m.w
      bP = m.b
      trainAcc[lambdas.indexOf(lam)] = accuracyOf(m, data)
    }
    let bestIdx = 0
    for (let i = 1; i < cvMean.length; i++) if (cvMean[i] > cvMean[bestIdx]) bestIdx = i
    return { cvMean, cvStd, trainAcc, bestIdx }
  }, [data, lambdas, k])

  const lx = (lam: number) =>
    PAD_L +
    ((Math.log10(lam) - Math.log10(LMIN)) /
      (Math.log10(LMAX) - Math.log10(LMIN))) *
      (W - PAD_L - PAD_R)
  const ly = (a: number) =>
    H - PAD_B - ((a - Y_MIN) / (Y_MAX - Y_MIN)) * (H - PAD_T - PAD_B)

  const cvPath = cvMean
    .map((a, i) => `${i === 0 ? 'M' : 'L'} ${lx(lambdas[i]).toFixed(1)} ${ly(a).toFixed(1)}`)
    .join(' ')
  const trainPath = trainAcc
    .map((a, i) => `${i === 0 ? 'M' : 'L'} ${lx(lambdas[i]).toFixed(1)} ${ly(a).toFixed(1)}`)
    .join(' ')

  const xTicks = [-3, -2, -1, 0]
  const expChar: Record<number, string> = { [-3]: '⁻³', [-2]: '⁻²', [-1]: '⁻¹', [0]: '⁰' }

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Folds k = <span className="font-mono">{k}</span>
          </span>
          <input
            type="range"
            min={3}
            max={10}
            step={1}
            value={k}
            onChange={(e) => setK(parseInt(e.target.value))}
            className="w-36"
          />
        </label>
        <label className="flex items-center gap-2 text-ink-muted">
          <input
            type="checkbox"
            checked={showTrain}
            onChange={(e) => setShowTrain(e.target.checked)}
          />
          Show training accuracy
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line x1={PAD_L} y1={ly(Y_MIN)} x2={W - PAD_R} y2={ly(Y_MIN)} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={ly(Y_MIN)} stroke="var(--rule)" strokeWidth={1} />

          {/* chosen-λ marker */}
          <line
            x1={lx(lambdas[bestIdx])}
            y1={PAD_T}
            x2={lx(lambdas[bestIdx])}
            y2={ly(Y_MIN)}
            stroke="var(--accent)"
            strokeWidth={1.25}
            strokeDasharray="3,3"
            opacity={0.6}
          />
          <text x={lx(lambdas[bestIdx])} y={PAD_T - 4} fontSize={10} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">
            best λ
          </text>

          {/* training accuracy overlay */}
          {showTrain && (
            <path d={trainPath} fill="none" stroke={TRAIN_COLOR} strokeWidth={1.75} strokeDasharray="4,3" />
          )}

          {/* CV error bars */}
          {cvMean.map((a, i) => (
            <line
              key={i}
              x1={lx(lambdas[i])}
              y1={ly(Math.max(Y_MIN, a - cvStd[i]))}
              x2={lx(lambdas[i])}
              y2={ly(Math.min(Y_MAX, a + cvStd[i]))}
              stroke={CV_COLOR}
              strokeWidth={1}
              opacity={0.4}
            />
          ))}

          {/* CV mean curve */}
          <path d={cvPath} fill="none" stroke={CV_COLOR} strokeWidth={2} />
          {cvMean.map((a, i) => (
            <circle key={i} cx={lx(lambdas[i])} cy={ly(a)} r={i === bestIdx ? 5 : 3} fill={CV_COLOR} stroke="var(--paper)" strokeWidth={1.1} />
          ))}

          {/* y ticks */}
          {[0.25, 0.5, 0.75, 1.0].map((t) => (
            <text key={t} x={PAD_L - 6} y={ly(t) + 3} fontSize={10} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">
              {t.toFixed(2)}
            </text>
          ))}
          {/* x ticks (log) */}
          {xTicks.map((e) => (
            <text key={e} x={lx(Math.pow(10, e))} y={ly(Y_MIN) + 16} fontSize={10} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">
              10{expChar[e]}
            </text>
          ))}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 6} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">
            regularisation strength λ
          </text>
          <text x={PAD_L} y={PAD_T - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">
            accuracy
          </text>

          {/* legend */}
          <g transform={`translate(${W - PAD_R - 132}, ${ly(Y_MIN) - 34})`}>
            <line x1={0} y1={0} x2={16} y2={0} stroke={CV_COLOR} strokeWidth={2} />
            <text x={20} y={3.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">CV accuracy</text>
            {showTrain && (
              <>
                <line x1={0} y1={15} x2={16} y2={15} stroke={TRAIN_COLOR} strokeWidth={1.75} strokeDasharray="4,3" />
                <text x={20} y={18.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">training accuracy</text>
              </>
            )}
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          best λ = <span className="text-accent">{lambdas[bestIdx].toExponential(1)}</span>
        </div>
        <div>
          CV at best = <span className="text-ink">{cvMean[bestIdx].toFixed(3)} ± {cvStd[bestIdx].toFixed(3)}</span>
        </div>
        <div>
          train at λ→0 = <span className="text-ink">{trainAcc[0].toFixed(3)}</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 11.3 — Choosing λ for L2 logistic regression by k-fold
        cross-validation, on data with three signal features buried among
        twenty noise features. Training accuracy (grey, dashed) rises toward a
        perfect score as λ → 0 — the model memorising the noise. The
        cross-validated accuracy (teal, with per-fold error bars) tells the
        truth: it peaks at a non-zero λ and falls off on both sides. The dial
        the training score would choose (λ = 0) is exactly the wrong one.
      </figcaption>
    </figure>
  )
}
