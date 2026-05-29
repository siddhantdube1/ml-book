'use client'

import { useMemo, useRef, useState } from 'react'
import {
  makeScores,
  confusionAt,
  accuracy,
  precision,
  recall,
  f1,
  type Scored,
} from '@/lib/metrics'

const W = 640
const H = 300
const PAD_X = 36
const PAD_TOP = 24
const PAD_BOT = 36
const BASE = (PAD_TOP + (H - PAD_BOT)) / 2 // baseline between the two histograms
const N_BINS = 30
const N_SAMPLES = 500

const NEG = '#3c5a8c' // class 0
const POS = '#c7522a' // class 1

const DATASETS = {
  balanced: 'Balanced (≈50% positive)',
  imbalanced: 'Imbalanced (≈5% positive)',
} as const
type DatasetKey = keyof typeof DATASETS

function generate(key: DatasetKey): Scored[] {
  switch (key) {
    case 'balanced':
      return makeScores(N_SAMPLES, 11, { baseRate: 0.5, spread: 1.8 })
    case 'imbalanced':
      return makeScores(N_SAMPLES, 23, { baseRate: 0.015, spread: 1.8 })
  }
}

export default function ThresholdConfusion() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('balanced')
  const [threshold, setThreshold] = useState(0.5)
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const data = useMemo(() => generate(datasetKey), [datasetKey])

  const sx = (s: number) => PAD_X + s * (W - 2 * PAD_X)
  const invX = (px: number) =>
    Math.max(0, Math.min(1, (px - PAD_X) / (W - 2 * PAD_X)))

  // Histograms: positives drawn upward from the baseline, negatives down.
  const { bins, maxCount } = useMemo(() => {
    const pos = new Array(N_BINS).fill(0)
    const neg = new Array(N_BINS).fill(0)
    for (const d of data) {
      const b = Math.min(N_BINS - 1, Math.floor(d.score * N_BINS))
      if (d.label === 1) pos[b]++
      else neg[b]++
    }
    let mx = 1
    for (let i = 0; i < N_BINS; i++) mx = Math.max(mx, pos[i], neg[i])
    return { bins: { pos, neg }, maxCount: mx }
  }, [data])

  const barW = (W - 2 * PAD_X) / N_BINS
  const halfH = BASE - PAD_TOP
  const barH = (count: number) => (count / maxCount) * halfH

  const c = useMemo(() => confusionAt(data, threshold), [data, threshold])
  const acc = accuracy(c)
  const prec = precision(c)
  const rec = recall(c)
  const f = f1(c)
  const posRate = useMemo(
    () => data.reduce((s, d) => s + d.label, 0) / data.length,
    [data],
  )

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!dragging) return
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { x } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    setThreshold(invX(x))
  }

  const tx = sx(threshold)

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Data</span>
          <select
            value={datasetKey}
            onChange={(e) => setDatasetKey(e.target.value as DatasetKey)}
            className="bg-paper border border-rule rounded px-2 py-1 text-ink"
          >
            {Object.entries(DATASETS).map(([k, label]) => (
              <option key={k} value={k}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Threshold = <span className="font-mono">{threshold.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-48"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block select-none"
          style={{ touchAction: 'none' }}
          onPointerMove={onPointerMove}
          onPointerUp={(e) => {
            setDragging(false)
            e.currentTarget.releasePointerCapture(e.pointerId)
          }}
        >
          {/* predicted-positive region shading (score >= threshold) */}
          <rect
            x={tx}
            y={PAD_TOP}
            width={W - PAD_X - tx}
            height={H - PAD_BOT - PAD_TOP}
            fill="var(--accent)"
            opacity={0.05}
          />

          {/* baseline */}
          <line
            x1={PAD_X}
            y1={BASE}
            x2={W - PAD_X}
            y2={BASE}
            stroke="var(--rule)"
            strokeWidth={1}
          />

          {/* positive bars (up) and negative bars (down) */}
          {bins.pos.map((count, i) => {
            const x = PAD_X + i * barW
            const h = barH(count)
            return count > 0 ? (
              <rect
                key={`p${i}`}
                x={x + 0.5}
                y={BASE - h}
                width={barW - 1}
                height={h}
                fill={POS}
                opacity={0.75}
              />
            ) : null
          })}
          {bins.neg.map((count, i) => {
            const x = PAD_X + i * barW
            const h = barH(count)
            return count > 0 ? (
              <rect
                key={`n${i}`}
                x={x + 0.5}
                y={BASE}
                width={barW - 1}
                height={h}
                fill={NEG}
                opacity={0.75}
              />
            ) : null
          })}

          {/* class labels */}
          <text
            x={PAD_X + 2}
            y={PAD_TOP + 4}
            fontSize={11}
            fill={POS}
            fontFamily="var(--font-sans, sans-serif)"
          >
            positives (label 1)
          </text>
          <text
            x={PAD_X + 2}
            y={H - PAD_BOT - 4}
            fontSize={11}
            fill={NEG}
            fontFamily="var(--font-sans, sans-serif)"
          >
            negatives (label 0)
          </text>

          {/* score axis ticks */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <text
              key={t}
              x={sx(t)}
              y={H - PAD_BOT + 16}
              fontSize={10}
              fill="var(--ink-muted)"
              textAnchor="middle"
              fontFamily="var(--font-mono, monospace)"
            >
              {t.toFixed(2)}
            </text>
          ))}

          {/* threshold line + draggable handle */}
          <line
            x1={tx}
            y1={PAD_TOP - 6}
            x2={tx}
            y2={H - PAD_BOT}
            stroke="var(--ink)"
            strokeWidth={1.5}
          />
          <rect
            x={tx - 7}
            y={PAD_TOP - 16}
            width={14}
            height={12}
            rx={2}
            fill="var(--ink)"
            className="cursor-ew-resize"
            onPointerDown={(e) => {
              setDragging(true)
              e.currentTarget.setPointerCapture(e.pointerId)
              e.stopPropagation()
            }}
          />
          {/* wide invisible grab strip so the line is easy to catch */}
          <rect
            x={tx - 10}
            y={PAD_TOP - 16}
            width={20}
            height={H - PAD_BOT - PAD_TOP + 16}
            fill="transparent"
            className="cursor-ew-resize"
            onPointerDown={(e) => {
              setDragging(true)
              e.currentTarget.setPointerCapture(e.pointerId)
              e.stopPropagation()
            }}
          />
          {/* static axis legend: lower scores → predict 0, higher → predict 1 */}
          <text
            x={(PAD_X + W - PAD_X) / 2}
            y={H - 4}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="middle"
            fontFamily="var(--font-sans, sans-serif)"
          >
            ← predict 0 (lower score)      predict 1 (higher score) →
          </text>
        </svg>
      </div>

      {/* Confusion matrix + metrics */}
      <div className="flex flex-col md:flex-row gap-6 mt-5 items-start">
        <div className="font-sans text-sm">
          <div
            className="grid"
            style={{
              gridTemplateColumns: 'auto 1fr 1fr',
              gridTemplateRows: 'auto auto auto',
            }}
          >
            <div />
            <div className="text-center text-ink-muted text-xs pb-1">
              predicted 1
            </div>
            <div className="text-center text-ink-muted text-xs pb-1">
              predicted 0
            </div>

            <div className="flex items-center text-ink-muted text-xs pr-2">
              actual 1
            </div>
            <ConfCell label="TP" value={c.tp} correct />
            <ConfCell label="FN" value={c.fn} />

            <div className="flex items-center text-ink-muted text-xs pr-2">
              actual 0
            </div>
            <ConfCell label="FP" value={c.fp} />
            <ConfCell label="TN" value={c.tn} correct />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-8 gap-y-2 font-mono text-sm">
          <Metric name="accuracy" value={acc} highlight />
          <Metric name="precision" value={prec} />
          <Metric name="recall" value={rec} />
          <Metric name="F1" value={f} />
          <div className="col-span-2 text-xs text-ink-muted pt-1">
            positive rate in data ={' '}
            <span className="font-mono text-ink">
              {(posRate * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-5 text-center max-w-prose mx-auto">
        Figure 10.1 — Predicted scores for {N_SAMPLES} examples: positives
        (orange, above) and negatives (blue, below). Drag the threshold and
        watch the confusion matrix and metrics update. Precision and recall
        move in opposite directions as you slide it. Switch to the
        imbalanced data — the threshold stays put — and notice accuracy
        stay high while precision and recall fall apart.
      </figcaption>
    </figure>
  )
}

function ConfCell({
  label,
  value,
  correct = false,
}: {
  label: string
  value: number
  correct?: boolean
}) {
  return (
    <div
      className="border border-rule m-0.5 px-4 py-3 text-center"
      style={{
        background: correct
          ? 'color-mix(in srgb, var(--accent) 14%, transparent)'
          : 'color-mix(in srgb, #c7522a 14%, transparent)',
      }}
    >
      <div className="font-mono text-lg text-ink">{value}</div>
      <div className="text-xs text-ink-muted">{label}</div>
    </div>
  )
}

function Metric({
  name,
  value,
  highlight = false,
}: {
  name: string
  value: number
  highlight?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-ink-muted">{name}</span>
      <span className={highlight ? 'text-accent' : 'text-ink'}>
        {value.toFixed(3)}
      </span>
    </div>
  )
}
