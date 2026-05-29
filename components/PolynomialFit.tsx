'use client'

import { useMemo, useState } from 'react'
import {
  polyFeatures,
  ridgeRegression,
  evalPoly,
  trueCurve,
  makeNoisyScatter,
} from '@/lib/regularisation'

const W = 640
const H = 360
const PAD_L = 36
const PAD_R = 24
const PAD_T = 24
const PAD_B = 36

const X_MIN = -1
const X_MAX = 1
const Y_MIN = -1.4
const Y_MAX = 1.4

const LOG_LAMBDA_MIN = -5
const LOG_LAMBDA_MAX = 1
const N_POINTS = 25

export default function PolynomialFit() {
  const [degree, setDegree] = useState(12)
  const [logLambda, setLogLambda] = useState(-4)
  const [seed, setSeed] = useState(3)
  const [showTrue, setShowTrue] = useState(true)

  const data = useMemo(() => makeNoisyScatter(N_POINTS, seed), [seed])

  const lambda = Math.pow(10, logLambda)

  const { w, trainMse, testMse } = useMemo(() => {
    const X = polyFeatures(
      data.map((p) => p.x),
      degree,
    )
    const y = data.map((p) => p.y)
    const wFit = ridgeRegression(X, y, lambda)
    let tr = 0
    for (let i = 0; i < data.length; i++) {
      const pred = evalPoly(wFit, data[i].x)
      const err = pred - data[i].y
      tr += err * err
    }
    // "Test" MSE: error against the noise-free true curve, sampled on a
    // dense grid. Measures how well the model recovers the true function.
    let te = 0
    const N = 200
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      const pred = evalPoly(wFit, x)
      const tru = trueCurve(x)
      te += (pred - tru) * (pred - tru)
    }
    return { w: wFit, trainMse: tr / data.length, testMse: te / (N + 1) }
  }, [data, degree, lambda])

  const sx = (x: number) =>
    PAD_L + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - PAD_L - PAD_R)
  const sy = (y: number) =>
    H - PAD_B - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - PAD_T - PAD_B)

  const clampY = (y: number) => Math.max(-50, Math.min(50, y))

  const fitPath = useMemo(() => {
    const N = 320
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      const y = clampY(evalPoly(w, x))
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(2)} ${sy(y).toFixed(2)}`)
    }
    return parts.join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w])

  const truePath = useMemo(() => {
    const N = 240
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      const y = trueCurve(x)
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(2)} ${sy(y).toFixed(2)}`)
    }
    return parts.join(' ')
  }, [])

  const formatLambda = (l: number): string => {
    if (l >= 1) return l.toFixed(2)
    if (l >= 0.01) return l.toFixed(3)
    return l.toExponential(1)
  }

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Degree = <span className="font-mono">{degree}</span>
          </span>
          <input
            type="range"
            min={1}
            max={15}
            step={1}
            value={degree}
            onChange={(e) => setDegree(parseInt(e.target.value))}
            className="w-32"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            λ = <span className="font-mono">{formatLambda(lambda)}</span>
          </span>
          <input
            type="range"
            min={LOG_LAMBDA_MIN}
            max={LOG_LAMBDA_MAX}
            step={0.05}
            value={logLambda}
            onChange={(e) => setLogLambda(parseFloat(e.target.value))}
            className="w-36"
          />
        </label>
        <label className="flex items-center gap-2 text-ink-muted">
          <input
            type="checkbox"
            checked={showTrue}
            onChange={(e) => setShowTrue(e.target.checked)}
          />
          Show true curve
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          New noise
        </button>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line
            x1={sx(X_MIN)}
            y1={sy(0)}
            x2={sx(X_MAX)}
            y2={sy(0)}
            stroke="var(--rule)"
            strokeWidth={1}
          />
          <line
            x1={sx(0)}
            y1={sy(Y_MIN)}
            x2={sx(0)}
            y2={sy(Y_MAX)}
            stroke="var(--rule)"
            strokeWidth={1}
            strokeDasharray="2,3"
          />

          {/* y ticks */}
          {[-1, 1].map((tick) => (
            <g key={tick}>
              <line
                x1={sx(X_MIN) - 3}
                y1={sy(tick)}
                x2={sx(X_MIN)}
                y2={sy(tick)}
                stroke="var(--ink-muted)"
                strokeWidth={1}
              />
              <text
                x={sx(X_MIN) - 6}
                y={sy(tick) + 3}
                fontSize={10}
                fill="var(--ink-muted)"
                textAnchor="end"
                fontFamily="var(--font-mono, monospace)"
              >
                {tick}
              </text>
            </g>
          ))}

          {/* x ticks */}
          {[-1, 0, 1].map((tick) => (
            <g key={tick}>
              <line
                x1={sx(tick)}
                y1={sy(0) + 3}
                x2={sx(tick)}
                y2={sy(0)}
                stroke="var(--ink-muted)"
                strokeWidth={1}
              />
              <text
                x={sx(tick)}
                y={sy(0) + 14}
                fontSize={10}
                fill="var(--ink-muted)"
                textAnchor="middle"
                fontFamily="var(--font-mono, monospace)"
              >
                {tick}
              </text>
            </g>
          ))}

          {/* true curve, subtle */}
          {showTrue && (
            <path
              d={truePath}
              fill="none"
              stroke="var(--ink-muted)"
              strokeWidth={1.5}
              strokeDasharray="4,3"
              opacity={0.7}
            />
          )}

          {/* fitted curve */}
          <path
            d={fitPath}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={2}
          />

          {/* scatter points */}
          {data.map((p, i) => (
            <circle
              key={i}
              cx={sx(p.x)}
              cy={sy(p.y)}
              r={4}
              fill="var(--ink)"
              stroke="var(--paper)"
              strokeWidth={1}
            />
          ))}

          {/* axis labels */}
          <text
            x={sx(X_MAX) - 4}
            y={sy(0) - 6}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="end"
            fontStyle="italic"
          >
            x
          </text>
          <text
            x={sx(0) + 8}
            y={sy(Y_MAX) + 12}
            fontSize={11}
            fill="var(--ink-muted)"
            fontStyle="italic"
          >
            y
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-4 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>degree = {degree}</div>
        <div>λ = {formatLambda(lambda)}</div>
        <div>
          train MSE ={' '}
          <span className="text-ink">{trainMse.toFixed(4)}</span>
        </div>
        <div>
          test MSE ={' '}
          <span className="text-ink">{testMse.toFixed(4)}</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 9.1 — Polynomial regression on 25 noisy samples (black) of a
        smooth underlying curve (dashed). The orange line is the ridge-
        regression fit at the chosen degree and λ. Crank the degree up with
        λ near zero to see overfitting take hold; crank λ up to watch
        regularisation pull the fit back toward the true curve.
      </figcaption>
    </figure>
  )
}
