'use client'

import { useMemo, useState } from 'react'
import {
  makeScores,
  calibrationBins,
  expectedCalibrationError,
  aucFromRoc,
  rocCurve,
  type Distortion,
} from '@/lib/metrics'

const PLOT = 320
const PAD_L = 48
const PAD_T = 20
const PAD_R = 20
const PAD_B = 44
const W = PAD_L + PLOT + PAD_R
const H = PAD_T + PLOT + PAD_B
const N_SAMPLES = 2000
const N_BINS = 10
const SEED = 31

const MODELS = {
  none: 'Well-calibrated',
  overconfident: 'Overconfident',
} as const
type ModelKey = Extract<Distortion, 'none' | 'overconfident'>

export default function CalibrationDiagram() {
  const [model, setModel] = useState<ModelKey>('none')

  const data = useMemo(
    () =>
      makeScores(N_SAMPLES, SEED, {
        baseRate: 0.5,
        spread: 1.8,
        distortion: model,
      }),
    [model],
  )

  const bins = useMemo(() => calibrationBins(data, N_BINS), [data])
  const ece = useMemo(() => expectedCalibrationError(bins), [bins])
  const auc = useMemo(() => aucFromRoc(rocCurve(data)), [data])

  const maxCount = useMemo(
    () => Math.max(1, ...bins.map((b) => b.count)),
    [bins],
  )

  const px = (v: number) => PAD_L + v * PLOT
  const py = (v: number) => PAD_T + (1 - v) * PLOT

  const nonEmpty = bins.filter((b) => b.count > 0)
  const curvePath = nonEmpty
    .map(
      (b, i) =>
        `${i === 0 ? 'M' : 'L'} ${px(b.meanPredicted).toFixed(2)} ${py(b.observedFreq).toFixed(2)}`,
    )
    .join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <div className="flex items-center gap-2">
          <span className="text-ink-muted">Model</span>
          <div className="inline-flex rounded border border-rule overflow-hidden">
            {(Object.keys(MODELS) as ModelKey[]).map((k) => (
              <button
                key={k}
                onClick={() => setModel(k)}
                className="px-3 py-1 transition-colors border-l border-rule first:border-l-0"
                style={{
                  background: model === k ? 'var(--accent)' : 'transparent',
                  color: model === k ? 'var(--paper)' : 'var(--ink)',
                }}
              >
                {MODELS[k]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <rect
            x={PAD_L}
            y={PAD_T}
            width={PLOT}
            height={PLOT}
            fill="none"
            stroke="var(--rule)"
            strokeWidth={1}
          />

          {/* count histogram underlay (subtle) */}
          {bins.map((b, i) => {
            if (b.count === 0) return null
            const barW = PLOT / N_BINS
            const h = (b.count / maxCount) * (PLOT * 0.18)
            return (
              <rect
                key={`c${i}`}
                x={px(b.lo) + 1}
                y={py(0) - h}
                width={barW - 2}
                height={h}
                fill="var(--ink-muted)"
                opacity={0.12}
              />
            )
          })}

          {/* perfect-calibration diagonal */}
          <line
            x1={px(0)}
            y1={py(0)}
            x2={px(1)}
            y2={py(1)}
            stroke="var(--rule)"
            strokeWidth={1}
            strokeDasharray="4,3"
          />
          {/* label parallels the diagonal but sits in the lower-right
              triangle (y < x), which stays empty for both the calibrated
              curve (dots on the diagonal) and the overconfident one (its
              right-half dots ride well above this point). */}
          <text
            x={px(0.62)}
            y={py(0.4)}
            fontSize={10}
            fill="var(--ink-muted)"
            textAnchor="middle"
            fontStyle="italic"
            transform={`rotate(-45 ${px(0.62).toFixed(1)} ${py(0.4).toFixed(1)})`}
          >
            perfect calibration
          </text>

          {/* reliability curve */}
          <path d={curvePath} fill="none" stroke="var(--accent)" strokeWidth={2} />
          {nonEmpty.map((b, i) => (
            <circle
              key={`p${i}`}
              cx={px(b.meanPredicted)}
              cy={py(b.observedFreq)}
              r={3 + 6 * Math.sqrt(b.count / maxCount)}
              fill="var(--accent)"
              stroke="var(--paper)"
              strokeWidth={1.25}
              opacity={0.85}
            />
          ))}

          {/* axis ticks */}
          {[0, 0.5, 1].map((t) => (
            <text
              key={`x${t}`}
              x={px(t)}
              y={PAD_T + PLOT + 16}
              fontSize={10}
              fill="var(--ink-muted)"
              textAnchor="middle"
              fontFamily="var(--font-mono, monospace)"
            >
              {t.toFixed(1)}
            </text>
          ))}
          {[0, 0.5, 1].map((t) => (
            <text
              key={`y${t}`}
              x={PAD_L - 8}
              y={py(t) + 3}
              fontSize={10}
              fill="var(--ink-muted)"
              textAnchor="end"
              fontFamily="var(--font-mono, monospace)"
            >
              {t.toFixed(1)}
            </text>
          ))}
          <text
            x={PAD_L + PLOT / 2}
            y={H - 6}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="middle"
            fontStyle="italic"
          >
            mean predicted probability
          </text>
          <text
            x={14}
            y={PAD_T + PLOT / 2}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="middle"
            fontStyle="italic"
            transform={`rotate(-90 14 ${PAD_T + PLOT / 2})`}
          >
            observed frequency
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-sm">
        <div className="flex items-baseline justify-between gap-3">
          <span className="text-ink-muted">AUC (ranking)</span>
          <span className="text-ink">{auc.toFixed(3)}</span>
        </div>
        <div className="flex items-baseline justify-between gap-3">
          <span className="text-ink-muted">ECE (calibration)</span>
          <span className="text-accent">{ece.toFixed(3)}</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 10.3 — A reliability diagram. Each dot is a bin of predictions;
        its position is mean predicted probability (x) against the fraction
        that were actually positive (y). On the diagonal, the probabilities
        are honest. Toggle between the two models: the AUC is identical —
        they rank examples equally well — but the overconfident model's
        curve bows away from the diagonal and its ECE jumps. Ranking well
        and being trustworthy are not the same thing.
      </figcaption>
    </figure>
  )
}
