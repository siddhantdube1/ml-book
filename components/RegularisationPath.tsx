'use client'

import { useMemo, useState } from 'react'
import {
  makeSparseDataset,
  regularisationPath,
  logspace,
} from '@/lib/regularisation'

const W = 640
const H = 400
const PAD_L = 50
const PAD_R = 24
const PAD_T = 32
const PAD_B = 44

// Dataset
const N_SAMPLES = 200
const N_INFORMATIVE = 3
const N_NOISE = 3
const N_FEATURES = N_INFORMATIVE + N_NOISE
const SIGNAL = [1.4, -1.0, 0.7]
const SEED = 7

// Path
const N_LAMBDAS = 60
const LAMBDA_MIN = 1e-3
const LAMBDA_MAX = 100

const FEATURE_COLORS = [
  'var(--accent)',
  '#3c5a8c',
  '#c7522a',
  '#9c9890',
  '#9c9890',
  '#9c9890',
] as const
const FEATURE_DASH = ['', '', '', '3,3', '3,3', '3,3'] as const
const FEATURE_NAMES = ['w₀', 'w₁', 'w₂', 'w₃', 'w₄', 'w₅'] as const

function exponentChar(n: number): string {
  const map: Record<number, string> = {
    [-3]: '⁻³',
    [-2]: '⁻²',
    [-1]: '⁻¹',
    [0]: '⁰',
    [1]: '¹',
    [2]: '²',
  }
  return map[n] ?? String(n)
}

// Defensive ceiling on the y-axis: standardised, stably-fit coefficients
// stay within roughly ±5, so this never clips real data but guarantees a
// single pathological solve can't blow the whole axis out to ±1e200.
const Y_CEIL = 8

export default function RegularisationPath() {
  const [kind, setKind] = useState<'L1' | 'L2'>('L1')
  // Opens at exactly 3/6 nonzero under L1 — all three signal coefficients
  // retained, all three noise coefficients zeroed: the cleanest snapshot
  // of "L1 selects features".
  const [lambdaIdx, setLambdaIdx] = useState(23)

  const data = useMemo(
    () =>
      makeSparseDataset(N_SAMPLES, N_INFORMATIVE, N_NOISE, SIGNAL, SEED, 0.5),
    [],
  )

  const lambdas = useMemo(() => logspace(LAMBDA_MIN, LAMBDA_MAX, N_LAMBDAS), [])

  const path = useMemo(
    () =>
      regularisationPath(data, kind, lambdas, {
        lr: 0.4,
        stepsPerLambda: 250,
      }),
    [data, kind, lambdas],
  )

  const yRange = useMemo(() => {
    let mn = Infinity
    let mx = -Infinity
    for (const e of path) {
      for (const c of e.w) {
        if (!Number.isFinite(c)) continue // never let a stray solve set the scale
        if (c < mn) mn = c
        if (c > mx) mx = c
      }
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) {
      mn = -1
      mx = 1
    }
    // Clamp to a robust finite window as a final backstop.
    mn = Math.max(mn, -Y_CEIL)
    mx = Math.min(mx, Y_CEIL)
    const span = Math.max(0.2, mx - mn)
    const pad = span * 0.1
    return [mn - pad, mx + pad] as [number, number]
  }, [path])

  // Clamp a coefficient into the visible window so a single bad value can
  // never render a line shooting off the canvas.
  const clampY = (v: number) => {
    if (!Number.isFinite(v)) return yRange[0]
    return Math.max(yRange[0], Math.min(yRange[1], v))
  }

  const lx = (lam: number) => {
    const t =
      (Math.log10(lam) - Math.log10(LAMBDA_MIN)) /
      (Math.log10(LAMBDA_MAX) - Math.log10(LAMBDA_MIN))
    return PAD_L + t * (W - PAD_L - PAD_R)
  }
  const ly = (val: number) =>
    H -
    PAD_B -
    ((val - yRange[0]) / (yRange[1] - yRange[0])) * (H - PAD_T - PAD_B)

  const linePaths = useMemo(() => {
    const ps: string[] = []
    for (let f = 0; f < N_FEATURES; f++) {
      const parts: string[] = []
      for (let i = 0; i < path.length; i++) {
        const x = lx(path[i].lambda)
        const y = ly(clampY(path[i].w[f]))
        parts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`)
      }
      ps.push(parts.join(' '))
    }
    return ps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, yRange])

  const current = path[Math.min(lambdaIdx, path.length - 1)]
  const currentLambda = current.lambda
  const nonzeroCount = current.w.filter((c) => Math.abs(c) > 0.005).length

  const xTicks = [-3, -2, -1, 0, 1, 2]

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <div className="flex items-center gap-2">
          <span className="text-ink-muted">Penalty</span>
          <div className="inline-flex rounded border border-rule overflow-hidden">
            <button
              onClick={() => setKind('L1')}
              className="px-3 py-1 transition-colors"
              style={{
                background: kind === 'L1' ? 'var(--accent)' : 'transparent',
                color: kind === 'L1' ? 'var(--paper)' : 'var(--ink)',
              }}
            >
              L1
            </button>
            <button
              onClick={() => setKind('L2')}
              className="px-3 py-1 transition-colors border-l border-rule"
              style={{
                background: kind === 'L2' ? 'var(--accent)' : 'transparent',
                color: kind === 'L2' ? 'var(--paper)' : 'var(--ink)',
              }}
            >
              L2
            </button>
          </div>
        </div>
        <label className="flex items-center gap-2 flex-1 max-w-md">
          <span className="text-ink-muted whitespace-nowrap">
            λ ={' '}
            <span className="font-mono">{currentLambda.toExponential(1)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={path.length - 1}
            value={lambdaIdx}
            onChange={(e) => setLambdaIdx(parseInt(e.target.value))}
            className="flex-1"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line
            x1={PAD_L}
            y1={ly(yRange[0])}
            x2={W - PAD_R}
            y2={ly(yRange[0])}
            stroke="var(--rule)"
            strokeWidth={1}
          />
          <line
            x1={PAD_L}
            y1={PAD_T}
            x2={PAD_L}
            y2={ly(yRange[0])}
            stroke="var(--rule)"
            strokeWidth={1}
          />

          {/* zero line */}
          {yRange[0] < 0 && yRange[1] > 0 && (
            <line
              x1={PAD_L}
              y1={ly(0)}
              x2={W - PAD_R}
              y2={ly(0)}
              stroke="var(--rule)"
              strokeWidth={1}
              strokeDasharray="2,3"
            />
          )}

          {/* x ticks */}
          {xTicks.map((tk) => {
            const x = lx(Math.pow(10, tk))
            return (
              <g key={tk}>
                <line
                  x1={x}
                  y1={ly(yRange[0])}
                  x2={x}
                  y2={ly(yRange[0]) + 4}
                  stroke="var(--ink-muted)"
                  strokeWidth={1}
                />
                <text
                  x={x}
                  y={ly(yRange[0]) + 16}
                  fontSize={10}
                  fill="var(--ink-muted)"
                  textAnchor="middle"
                  fontFamily="var(--font-mono, monospace)"
                >
                  10{exponentChar(tk)}
                </text>
              </g>
            )
          })}

          {/* y ticks */}
          {[yRange[0], 0, yRange[1]]
            .filter((v) => v >= yRange[0] && v <= yRange[1])
            .map((v, i) => (
              <text
                key={i}
                x={PAD_L - 6}
                y={ly(v) + 3}
                fontSize={10}
                fill="var(--ink-muted)"
                textAnchor="end"
                fontFamily="var(--font-mono, monospace)"
              >
                {v.toFixed(1)}
              </text>
            ))}

          {/* axis labels */}
          <text
            x={(PAD_L + W - PAD_R) / 2}
            y={H - 6}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="middle"
            fontStyle="italic"
          >
            regularisation strength λ
          </text>
          <text
            x={PAD_L}
            y={PAD_T - 10}
            fontSize={11}
            fill="var(--ink-muted)"
            fontStyle="italic"
          >
            coefficient value
          </text>

          {/* coefficient paths */}
          {linePaths.map((d, f) => (
            <path
              key={f}
              d={d}
              fill="none"
              stroke={FEATURE_COLORS[f]}
              strokeWidth={f < N_INFORMATIVE ? 2 : 1.5}
              strokeDasharray={FEATURE_DASH[f] || undefined}
              opacity={f < N_INFORMATIVE ? 0.9 : 0.6}
            />
          ))}

          {/* current λ vertical line */}
          <line
            x1={lx(currentLambda)}
            y1={PAD_T}
            x2={lx(currentLambda)}
            y2={ly(yRange[0])}
            stroke="var(--ink)"
            strokeWidth={1.25}
            opacity={0.55}
          />

          {/* dots at current λ */}
          {current.w.map((c, f) => (
            <circle
              key={f}
              cx={lx(currentLambda)}
              cy={ly(clampY(c))}
              r={4}
              fill={FEATURE_COLORS[f]}
              stroke="var(--paper)"
              strokeWidth={1.25}
            />
          ))}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-5 gap-y-1 mt-3 font-mono text-xs text-ink-muted justify-center">
        {FEATURE_NAMES.map((n, f) => (
          <div key={f} className="flex items-center gap-1.5">
            <svg width={20} height={6} className="inline-block">
              <line
                x1={0}
                y1={3}
                x2={20}
                y2={3}
                stroke={FEATURE_COLORS[f]}
                strokeWidth={2}
                strokeDasharray={FEATURE_DASH[f] || undefined}
              />
            </svg>
            {n} {f < N_INFORMATIVE ? '(signal)' : '(noise)'} ={' '}
            <span className="text-ink">{current.w[f].toFixed(3)}</span>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 mt-2 font-mono text-xs text-ink-muted">
        <div>λ = {currentLambda.toExponential(2)}</div>
        <div>
          nonzero coefficients ={' '}
          <span className="text-ink">
            {nonzeroCount} / {N_FEATURES}
          </span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 9.3 — The regularisation path. Six features — three carry
        signal (solid lines) and three are pure noise (dashed grey) — fit
        by logistic regression with an L1 or L2 penalty at sixty values
        of λ (log-spaced). Move the slider to change λ and watch every
        coefficient's value. Under L1 the noise coefficients hit zero
        first; under L2 every coefficient shrinks smoothly together, none
        ever quite reaching zero.
      </figcaption>
    </figure>
  )
}
