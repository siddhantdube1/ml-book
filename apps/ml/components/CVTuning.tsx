'use client'

import { useMemo, useState } from 'react'
import {
  makeNoisyScatter,
  polyFeatures,
  ridgeRegression,
  evalPoly,
} from '@/lib/regularisation'
import { kFoldSplit } from '@/lib/crossval'

const W = 640
const H = 380
const PAD_L = 50
const PAD_R = 20
const PAD_T = 24
const PAD_B = 46

const N = 60
const DEGREE = 12
const N_LAMBDA = 15
const LMIN = 1e-5
const LMAX = 1

const CV_COLOR = 'var(--accent)'
const TRAIN_COLOR = 'var(--ink-muted)'

const SUP: Record<string, string> = {
  '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³',
  '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
}
const sup = (n: number) =>
  String(n)
    .split('')
    .map((c) => SUP[c] ?? c)
    .join('')

function mse(w: number[], xs: number[], ys: number[]): number {
  let s = 0
  for (let i = 0; i < xs.length; i++) {
    const e = evalPoly(w, xs[i]) - ys[i]
    s += e * e
  }
  return s / xs.length
}

export default function CVTuning() {
  const [k, setK] = useState(5)
  const [showTrain, setShowTrain] = useState(true)

  const data = useMemo(() => makeNoisyScatter(N, 3, 0.18), [])
  const lambdas = useMemo(() => {
    const out: number[] = []
    const lo = Math.log10(LMIN)
    const hi = Math.log10(LMAX)
    for (let i = 0; i < N_LAMBDA; i++)
      out.push(Math.pow(10, lo + (i / (N_LAMBDA - 1)) * (hi - lo)))
    return out
  }, [])

  const { cvMean, cvStd, trainErr, bestIdx, yMax } = useMemo(() => {
    const folds = kFoldSplit(N, k, 1)
    const cvMean: number[] = []
    const cvStd: number[] = []
    for (const lam of lambdas) {
      const fold = folds.map((f) => {
        const tx = f.trainIdx.map((i) => data[i].x)
        const ty = f.trainIdx.map((i) => data[i].y)
        const ex = f.testIdx.map((i) => data[i].x)
        const ey = f.testIdx.map((i) => data[i].y)
        const w = ridgeRegression(polyFeatures(tx, DEGREE), ty, lam)
        return mse(w, ex, ey)
      })
      const m = fold.reduce((a, b) => a + b, 0) / fold.length
      const sd = Math.sqrt(
        fold.reduce((a, b) => a + (b - m) * (b - m), 0) / fold.length,
      )
      cvMean.push(m)
      cvStd.push(sd)
    }
    // Training error: fit on all data, score on all data.
    const allX = data.map((d) => d.x)
    const allY = data.map((d) => d.y)
    const trainErr = lambdas.map((lam) =>
      mse(ridgeRegression(polyFeatures(allX, DEGREE), allY, lam), allX, allY),
    )
    let bestIdx = 0
    for (let i = 1; i < cvMean.length; i++)
      if (cvMean[i] < cvMean[bestIdx]) bestIdx = i
    const yMax =
      Math.min(
        0.25,
        Math.max(...cvMean.map((m, i) => m + cvStd[i]), ...trainErr) * 1.08,
      )
    return { cvMean, cvStd, trainErr, bestIdx, yMax }
  }, [data, lambdas, k])

  const lx = (lam: number) =>
    PAD_L +
    ((Math.log10(lam) - Math.log10(LMIN)) /
      (Math.log10(LMAX) - Math.log10(LMIN))) *
      (W - PAD_L - PAD_R)
  const ly = (e: number) =>
    H - PAD_B - (Math.min(e, yMax) / yMax) * (H - PAD_T - PAD_B)

  const cvPath = cvMean
    .map((e, i) => `${i === 0 ? 'M' : 'L'} ${lx(lambdas[i]).toFixed(1)} ${ly(e).toFixed(1)}`)
    .join(' ')
  const trainPath = trainErr
    .map((e, i) => `${i === 0 ? 'M' : 'L'} ${lx(lambdas[i]).toFixed(1)} ${ly(e).toFixed(1)}`)
    .join(' ')

  const xTicks = [-5, -4, -3, -2, -1, 0]

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
          Show training error
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line x1={PAD_L} y1={ly(0)} x2={W - PAD_R} y2={ly(0)} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={ly(0)} stroke="var(--rule)" strokeWidth={1} />

          {/* chosen-λ marker */}
          <line
            x1={lx(lambdas[bestIdx])}
            y1={PAD_T}
            x2={lx(lambdas[bestIdx])}
            y2={ly(0)}
            stroke="var(--accent)"
            strokeWidth={1.25}
            strokeDasharray="3,3"
            opacity={0.6}
          />
          {(() => {
            const mx = lx(lambdas[bestIdx])
            const nearLeft = mx < PAD_L + 26
            return (
              <text
                x={nearLeft ? mx + 3 : mx}
                y={PAD_T - 4}
                fontSize={10}
                fill="var(--accent)"
                textAnchor={nearLeft ? 'start' : 'middle'}
                fontFamily="var(--font-mono, monospace)"
              >
                best λ
              </text>
            )
          })()}

          {/* training error overlay */}
          {showTrain && (
            <path d={trainPath} fill="none" stroke={TRAIN_COLOR} strokeWidth={1.75} strokeDasharray="4,3" />
          )}

          {/* CV error bars */}
          {cvMean.map((e, i) => (
            <line
              key={i}
              x1={lx(lambdas[i])}
              y1={ly(Math.max(0, e - cvStd[i]))}
              x2={lx(lambdas[i])}
              y2={ly(Math.min(yMax, e + cvStd[i]))}
              stroke={CV_COLOR}
              strokeWidth={1}
              opacity={0.4}
            />
          ))}

          {/* CV error curve */}
          <path d={cvPath} fill="none" stroke={CV_COLOR} strokeWidth={2} />
          {cvMean.map((e, i) => (
            <circle key={i} cx={lx(lambdas[i])} cy={ly(e)} r={i === bestIdx ? 5 : 3} fill={CV_COLOR} stroke="var(--paper)" strokeWidth={1.1} />
          ))}

          {/* y ticks */}
          {[0, yMax / 2, yMax].map((t, i) => (
            <text key={i} x={PAD_L - 6} y={ly(t) + 3} fontSize={10} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">
              {t.toFixed(2)}
            </text>
          ))}
          {/* x ticks (log) */}
          {xTicks.map((e) => (
            <text key={e} x={lx(Math.pow(10, e))} y={ly(0) + 16} fontSize={10} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">
              10{sup(e)}
            </text>
          ))}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 6} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">
            ridge penalty λ
          </text>
          <text
            x={14}
            y={(PAD_T + ly(0)) / 2}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="middle"
            fontStyle="italic"
            transform={`rotate(-90 14 ${(PAD_T + ly(0)) / 2})`}
          >
            error (MSE)
          </text>

          {/* legend */}
          <g transform={`translate(${W - PAD_R - 124}, ${PAD_T + 4})`}>
            <line x1={0} y1={0} x2={16} y2={0} stroke={CV_COLOR} strokeWidth={2} />
            <text x={20} y={3.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">CV error</text>
            {showTrain && (
              <>
                <line x1={0} y1={15} x2={16} y2={15} stroke={TRAIN_COLOR} strokeWidth={1.75} strokeDasharray="4,3" />
                <text x={20} y={18.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">training error</text>
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
          CV error at best = <span className="text-ink">{cvMean[bestIdx].toFixed(4)}</span>
        </div>
        <div>
          train error at λ→0 = <span className="text-ink">{trainErr[0].toFixed(4)}</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 11.3 — Choosing the ridge penalty λ for a degree-12 polynomial
        by k-fold cross-validation — the same overfitting-prone fit as Figure
        11.1, now tuned by λ instead of degree. Training error (grey, dashed)
        sinks toward zero as λ → 0, the polynomial wriggling through the noise.
        The cross-validated error (teal, with per-fold error bars) traces a
        clean U: it is worst at λ → 0 (overfitting) and at large λ
        (over-smoothing), and lowest at a non-trivial λ in between. The
        training error would drive λ to zero; cross-validation finds the
        bottom of the U.
      </figcaption>
    </figure>
  )
}
