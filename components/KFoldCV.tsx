'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  makeSparseDataset,
  trainLogisticL2,
  type LRPoint,
} from '@/lib/regularisation'
import { kFoldSplit } from '@/lib/crossval'
import { sigmoid } from '@/lib/logistic'

const W = 640
const H = 300
const PAD_L = 44
const PAD_R = 20
const BAR_Y = 46
const BAR_H = 40
const AX_Y = 210 // score axis baseline
const AX_MIN = 0.5
const AX_MAX = 1.0

const N = 120
const LAMBDA = 0.05

const TRAIN_FILL = 'var(--rule)'
const TEST_FILL = 'var(--accent)'

function accuracyOf(model: { w: number[]; b: number }, pts: LRPoint[]): number {
  let c = 0
  for (const s of pts) {
    let z = model.b
    for (let j = 0; j < model.w.length; j++) z += model.w[j] * s.x[j]
    if ((sigmoid(z) >= 0.5 ? 1 : 0) === s.y) c++
  }
  return c / pts.length
}

export default function KFoldCV() {
  const [k, setK] = useState(5)
  const [foldIdx, setFoldIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const data = useMemo(
    () => makeSparseDataset(N, 3, 2, [1.5, -1.1, 0.8], 4, 0.5),
    [],
  )

  const { folds, foldScores, mean, std, lo, hi } = useMemo(() => {
    const folds = kFoldSplit(N, k, 2)
    const foldScores = folds.map((f) => {
      const train = f.trainIdx.map((i) => data[i])
      const model = trainLogisticL2(train, LAMBDA, 0.4, 300)
      return accuracyOf(model, f.testIdx.map((i) => data[i]))
    })
    const mean = foldScores.reduce((a, b) => a + b, 0) / foldScores.length
    const std = Math.sqrt(
      foldScores.reduce((a, b) => a + (b - mean) * (b - mean), 0) /
        foldScores.length,
    )
    return {
      folds,
      foldScores,
      mean,
      std,
      lo: Math.min(...foldScores),
      hi: Math.max(...foldScores),
    }
  }, [data, k])

  useEffect(() => {
    setFoldIdx(0)
    setPlaying(false)
  }, [k])

  useEffect(() => {
    if (!playing) return
    const t = setTimeout(() => setFoldIdx((i) => (i + 1) % k), 750)
    return () => clearTimeout(t)
  }, [playing, foldIdx, k])

  const ax = (s: number) =>
    PAD_L + ((s - AX_MIN) / (AX_MAX - AX_MIN)) * (W - PAD_L - PAD_R)

  // Fold band widths proportional to fold size (in shuffled order).
  const bandW = (W - PAD_L - PAD_R) / k

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Folds k = <span className="font-mono">{k}</span>
          </span>
          <input
            type="range"
            min={2}
            max={10}
            step={1}
            value={k}
            onChange={(e) => setK(parseInt(e.target.value))}
            className="w-40"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block select-none">
          {/* label */}
          <text x={PAD_L} y={26} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            the {N} examples, split into {k} folds — fold {foldIdx + 1} is held out for testing
          </text>

          {/* fold bands */}
          {folds.map((_, f) => {
            const x = PAD_L + f * bandW
            const held = f === foldIdx
            return (
              <g key={f}>
                <rect
                  x={x + 1}
                  y={BAR_Y}
                  width={bandW - 2}
                  height={BAR_H}
                  fill={held ? TEST_FILL : TRAIN_FILL}
                  opacity={held ? 0.5 : 0.35}
                  stroke={held ? TEST_FILL : 'var(--rule)'}
                  strokeWidth={held ? 1.5 : 1}
                />
                <text
                  x={x + bandW / 2}
                  y={BAR_Y + BAR_H / 2 + 4}
                  fontSize={10}
                  fill="var(--ink)"
                  textAnchor="middle"
                  fontFamily="var(--font-sans, sans-serif)"
                  opacity={held ? 1 : 0.6}
                >
                  {held ? 'test' : 'train'}
                </text>
              </g>
            )
          })}
          {/* held-out fold caption */}
          <text x={PAD_L} y={BAR_Y + BAR_H + 18} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            each fold is the test set exactly once; the rest train the model
          </text>

          {/* score axis */}
          <line x1={ax(AX_MIN)} y1={AX_Y} x2={ax(AX_MAX)} y2={AX_Y} stroke="var(--rule)" strokeWidth={1} />
          {[0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((t) => (
            <g key={t}>
              <line x1={ax(t)} y1={AX_Y} x2={ax(t)} y2={AX_Y + 4} stroke="var(--ink-muted)" strokeWidth={1} />
              <text x={ax(t)} y={AX_Y + 16} fontSize={9} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">
                {t.toFixed(1)}
              </text>
            </g>
          ))}
          <text x={(ax(AX_MIN) + ax(AX_MAX)) / 2} y={AX_Y + 34} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">
            held-out accuracy per fold
          </text>

          {/* ±std band + mean line */}
          <rect
            x={ax(Math.max(AX_MIN, mean - std))}
            y={AX_Y - 54}
            width={ax(Math.min(AX_MAX, mean + std)) - ax(Math.max(AX_MIN, mean - std))}
            height={54}
            fill="var(--accent)"
            opacity={0.08}
          />
          <line x1={ax(mean)} y1={AX_Y - 58} x2={ax(mean)} y2={AX_Y} stroke="var(--accent)" strokeWidth={1.5} />
          <text x={ax(mean)} y={AX_Y - 62} fontSize={10} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">
            CV mean {mean.toFixed(3)}
          </text>

          {/* per-fold score dots */}
          {foldScores.map((s, f) => {
            const held = f === foldIdx
            // stagger vertically a little so coincident scores don't fully overlap
            const yy = AX_Y - 16 - (f % 3) * 12
            return (
              <circle
                key={f}
                cx={ax(s)}
                cy={yy}
                r={held ? 6 : 4}
                fill={held ? TEST_FILL : 'var(--ink-muted)'}
                stroke="var(--paper)"
                strokeWidth={1.25}
                opacity={held ? 1 : 0.7}
              />
            )
          })}
        </svg>
      </div>

      {/* playback */}
      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[60px]"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => {
            setPlaying(false)
            setFoldIdx((i) => (i + 1) % k)
          }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Step fold
        </button>
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">
          fold {foldIdx + 1} / {k}: accuracy {foldScores[foldIdx].toFixed(3)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          CV estimate = <span className="text-accent">{mean.toFixed(3)} ± {std.toFixed(3)}</span>
        </div>
        <div>
          single-fold range = <span className="text-ink">{lo.toFixed(3)} – {hi.toFixed(3)}</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 11.2 — k-fold cross-validation. The data is partitioned into k
        folds; each fold is held out as the test set once while the other k−1
        train the model, giving k accuracy estimates. Step through the folds
        to see each one held out in turn. Any single fold lands anywhere in
        the range below; the average across all k is the stable estimate.
        Raise k and watch the spread of individual folds widen even as the
        mean barely moves.
      </figcaption>
    </figure>
  )
}
