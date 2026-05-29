'use client'

import { useMemo, useState } from 'react'
import {
  makeNoisyScatter,
  polyFeatures,
  ridgeRegression,
  evalPoly,
  trueCurve,
} from '@/lib/regularisation'
import { trainTestSplit } from '@/lib/crossval'

// Error-vs-degree panel
const EW = 380
const EH = 320
const E_PAD_L = 46
const E_PAD_R = 16
const E_PAD_T = 22
const E_PAD_B = 40

// Fit inset panel
const IW = 300
const IH = 320
const I_PAD = 26
const IX_MIN = -1
const IX_MAX = 1
const IY_MIN = -1.5
const IY_MAX = 1.5

const N = 70
const MAX_DEG = 15
const LAMBDA = 1e-6

const TRAIN_COLOR = 'var(--ink-muted)'
const TEST_COLOR = 'var(--accent)'

export default function OptimismGap() {
  const [degree, setDegree] = useState(7)

  // Fixed dataset + split (computed once).
  const { data, trainXs, trainYs, testXs, testYs, testSet } = useMemo(() => {
    const data = makeNoisyScatter(N, 3, 0.18)
    const split = trainTestSplit(N, 0.4, 7)
    return {
      data,
      trainXs: split.trainIdx.map((i) => data[i].x),
      trainYs: split.trainIdx.map((i) => data[i].y),
      testXs: split.testIdx.map((i) => data[i].x),
      testYs: split.testIdx.map((i) => data[i].y),
      testSet: new Set(split.testIdx),
    }
  }, [])

  const mse = (w: number[], xs: number[], ys: number[]) => {
    let s = 0
    for (let i = 0; i < xs.length; i++) {
      const e = evalPoly(w, xs[i]) - ys[i]
      s += e * e
    }
    return s / xs.length
  }

  // Train/test error for every degree (computed once).
  const { trainErr, testErr, sweetSpot, yMax } = useMemo(() => {
    const trainErr: number[] = []
    const testErr: number[] = []
    for (let d = 1; d <= MAX_DEG; d++) {
      const w = ridgeRegression(polyFeatures(trainXs, d), trainYs, LAMBDA)
      trainErr.push(mse(w, trainXs, trainYs))
      testErr.push(mse(w, testXs, testYs))
    }
    let sweetSpot = 1
    for (let d = 2; d <= MAX_DEG; d++) {
      if (testErr[d - 1] < testErr[sweetSpot - 1]) sweetSpot = d
    }
    const yMax = Math.max(...trainErr, ...testErr) * 1.1
    return { trainErr, testErr, sweetSpot, yMax }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainXs, trainYs, testXs, testYs])

  // Current fitted polynomial (for the inset).
  const wCur = useMemo(
    () => ridgeRegression(polyFeatures(trainXs, degree), trainYs, LAMBDA),
    [trainXs, trainYs, degree],
  )

  // ── Error-panel transforms ──
  const ex = (d: number) =>
    E_PAD_L + ((d - 1) / (MAX_DEG - 1)) * (EW - E_PAD_L - E_PAD_R)
  const ey = (m: number) =>
    EH - E_PAD_B - (Math.min(m, yMax) / yMax) * (EH - E_PAD_T - E_PAD_B)

  const trainPath = trainErr
    .map((m, i) => `${i === 0 ? 'M' : 'L'} ${ex(i + 1).toFixed(1)} ${ey(m).toFixed(1)}`)
    .join(' ')
  const testPath = testErr
    .map((m, i) => `${i === 0 ? 'M' : 'L'} ${ex(i + 1).toFixed(1)} ${ey(m).toFixed(1)}`)
    .join(' ')

  // ── Inset transforms ──
  const ix = (x: number) =>
    I_PAD + ((x - IX_MIN) / (IX_MAX - IX_MIN)) * (IW - 2 * I_PAD)
  const iy = (y: number) =>
    IH - I_PAD - ((y - IY_MIN) / (IY_MAX - IY_MIN)) * (IH - 2 * I_PAD)
  const clampYI = (y: number) => Math.max(IY_MIN, Math.min(IY_MAX, y))

  const fitPath = useMemo(() => {
    const Npts = 200
    const parts: string[] = []
    for (let i = 0; i <= Npts; i++) {
      const x = IX_MIN + (i / Npts) * (IX_MAX - IX_MIN)
      const y = clampYI(evalPoly(wCur, x))
      parts.push(`${i === 0 ? 'M' : 'L'} ${ix(x).toFixed(1)} ${iy(y).toFixed(1)}`)
    }
    return parts.join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wCur])

  const truePath = useMemo(() => {
    const Npts = 160
    const parts: string[] = []
    for (let i = 0; i <= Npts; i++) {
      const x = IX_MIN + (i / Npts) * (IX_MAX - IX_MIN)
      parts.push(`${i === 0 ? 'M' : 'L'} ${ix(x).toFixed(1)} ${iy(trueCurve(x)).toFixed(1)}`)
    }
    return parts.join(' ')
  }, [])

  const fitLabel =
    degree < sweetSpot - 1
      ? 'underfit'
      : degree > sweetSpot + 1
        ? 'overfit'
        : 'good fit'

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Polynomial degree = <span className="font-mono">{degree}</span>
          </span>
          <input
            type="range"
            min={1}
            max={MAX_DEG}
            step={1}
            value={degree}
            onChange={(e) => setDegree(parseInt(e.target.value))}
            className="w-48"
          />
        </label>
        <button
          onClick={() => setDegree(sweetSpot)}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Jump to sweet spot
        </button>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <div className="flex flex-col md:flex-row">
          {/* ERROR-VS-DEGREE PANEL */}
          <svg
            viewBox={`0 0 ${EW} ${EH}`}
            className="w-full md:w-[56%] h-auto block border-b md:border-b-0 md:border-r border-rule"
          >
            {/* axes */}
            <line x1={E_PAD_L} y1={ey(0)} x2={EW - E_PAD_R} y2={ey(0)} stroke="var(--rule)" strokeWidth={1} />
            <line x1={E_PAD_L} y1={E_PAD_T} x2={E_PAD_L} y2={ey(0)} stroke="var(--rule)" strokeWidth={1} />

            {/* sweet-spot marker */}
            <line
              x1={ex(sweetSpot)}
              y1={E_PAD_T}
              x2={ex(sweetSpot)}
              y2={ey(0)}
              stroke="var(--accent)"
              strokeWidth={1}
              strokeDasharray="2,3"
              opacity={0.5}
            />

            {/* current-degree marker */}
            <line
              x1={ex(degree)}
              y1={E_PAD_T}
              x2={ex(degree)}
              y2={ey(0)}
              stroke="var(--ink)"
              strokeWidth={1.25}
              opacity={0.4}
            />

            {/* curves */}
            <path d={trainPath} fill="none" stroke={TRAIN_COLOR} strokeWidth={1.75} strokeDasharray="4,3" />
            <path d={testPath} fill="none" stroke={TEST_COLOR} strokeWidth={2} />

            {/* dots at current degree */}
            <circle cx={ex(degree)} cy={ey(trainErr[degree - 1])} r={4} fill={TRAIN_COLOR} stroke="var(--paper)" strokeWidth={1.25} />
            <circle cx={ex(degree)} cy={ey(testErr[degree - 1])} r={4} fill={TEST_COLOR} stroke="var(--paper)" strokeWidth={1.25} />

            {/* x ticks */}
            {[1, 5, 10, 15].map((d) => (
              <text key={d} x={ex(d)} y={ey(0) + 16} fontSize={10} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">
                {d}
              </text>
            ))}
            <text x={(E_PAD_L + EW - E_PAD_R) / 2} y={EH - 4} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">
              model complexity (polynomial degree)
            </text>
            <text x={E_PAD_L} y={E_PAD_T - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">
              error (MSE)
            </text>

            {/* legend */}
            <g transform={`translate(${EW - E_PAD_R - 120}, ${E_PAD_T + 4})`}>
              <line x1={0} y1={0} x2={16} y2={0} stroke={TEST_COLOR} strokeWidth={2} />
              <text x={20} y={3.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">test error</text>
              <line x1={0} y1={15} x2={16} y2={15} stroke={TRAIN_COLOR} strokeWidth={1.75} strokeDasharray="4,3" />
              <text x={20} y={18.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">training error</text>
            </g>
          </svg>

          {/* FIT INSET PANEL */}
          <svg viewBox={`0 0 ${IW} ${IH}`} className="w-full md:w-[44%] h-auto block">
            {/* zero axis */}
            <line x1={ix(IX_MIN)} y1={iy(0)} x2={ix(IX_MAX)} y2={iy(0)} stroke="var(--rule)" strokeWidth={1} strokeDasharray="2,3" />

            {/* true curve */}
            <path d={truePath} fill="none" stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7} />

            {/* fitted curve */}
            <path d={fitPath} fill="none" stroke="var(--accent)" strokeWidth={2} />

            {/* train + test points */}
            {data.map((p, i) => {
              const isTest = testSet.has(i)
              return (
                <circle
                  key={i}
                  cx={ix(p.x)}
                  cy={iy(clampYI(p.y))}
                  r={isTest ? 3.5 : 3}
                  fill={isTest ? 'none' : 'var(--ink)'}
                  stroke={isTest ? 'var(--accent)' : 'var(--paper)'}
                  strokeWidth={isTest ? 1.5 : 0.8}
                  opacity={isTest ? 0.9 : 0.55}
                />
              )
            })}

            <text x={I_PAD} y={I_PAD - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">
              fit at degree {degree} — {fitLabel}
            </text>
          </svg>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          train MSE = <span className="text-ink">{trainErr[degree - 1].toFixed(4)}</span>
        </div>
        <div>
          test MSE ={' '}
          <span style={{ color: 'var(--accent)' }}>{testErr[degree - 1].toFixed(4)}</span>
        </div>
        <div>sweet spot = degree {sweetSpot}</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 11.1 — One train/test split of a noisy dataset. As polynomial
        degree grows, training error (grey, dashed) falls without limit, but
        the held-out test error (teal) bottoms out and turns back up. The gap
        between them is overfitting, measured. The inset shows the fit itself
        at the chosen degree; drag past the sweet spot and watch it start
        chasing the noise.
      </figcaption>
    </figure>
  )
}
