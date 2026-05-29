'use client'

import { useMemo, useState } from 'react'
import {
  makeScores,
  rocCurve,
  prCurve,
  aucFromRoc,
  averagePrecision,
  confusionAt,
  precision,
  recall,
  fpr,
  type Scored,
} from '@/lib/metrics'

const PLOT = 320
const PAD_L = 48
const PAD_T = 20
const PAD_R = 20
const PAD_B = 44
const W = PAD_L + PLOT + PAD_R
const H = PAD_T + PLOT + PAD_B
const N_SAMPLES = 500

const DATASETS = {
  balanced: 'Balanced',
  imbalanced: 'Imbalanced (≈5%)',
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

export default function RocCurve() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('balanced')
  const [view, setView] = useState<'ROC' | 'PR'>('ROC')
  const [threshold, setThreshold] = useState(0.5)

  const data = useMemo(() => generate(datasetKey), [datasetKey])
  const roc = useMemo(() => rocCurve(data), [data])
  const pr = useMemo(() => prCurve(data), [data])
  const auc = useMemo(() => aucFromRoc(roc), [roc])
  const ap = useMemo(() => averagePrecision(pr), [pr])
  const posRate = useMemo(
    () => data.reduce((s, d) => s + d.label, 0) / data.length,
    [data],
  )

  // Operating point at the current threshold — same data, same lib.
  const c = useMemo(() => confusionAt(data, threshold), [data, threshold])
  const markerX = view === 'ROC' ? fpr(c) : recall(c)
  const markerY = view === 'ROC' ? recall(c) : precision(c)

  const px = (v: number) => PAD_L + v * PLOT
  const py = (v: number) => PAD_T + (1 - v) * PLOT

  // PR curves carry a leading (recall=0, precision=1) anchor used only for
  // the average-precision area. Dropping it from the *drawn* paths avoids a
  // spurious segment from (0,1) across empty space to the first real point
  // (which, under heavy imbalance, sits at the origin — the top-scored
  // examples are false positives). ROC keeps its (0,0) anchor: that point
  // is the genuine bottom-left start of the curve.
  const prDrawn = useMemo(() => (pr.length > 1 ? pr.slice(1) : pr), [pr])

  const curvePath = useMemo(() => {
    if (view === 'ROC') {
      return roc
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${px(p.fpr).toFixed(2)} ${py(p.tpr).toFixed(2)}`)
        .join(' ')
    }
    return prDrawn
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${px(p.recall).toFixed(2)} ${py(p.precision).toFixed(2)}`)
      .join(' ')
  }, [view, roc, prDrawn])

  const fillPath = useMemo(() => {
    if (view === 'ROC') {
      const head = roc
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${px(p.fpr).toFixed(2)} ${py(p.tpr).toFixed(2)}`)
        .join(' ')
      return `${head} L ${px(1).toFixed(2)} ${py(0).toFixed(2)} L ${px(0).toFixed(2)} ${py(0).toFixed(2)} Z`
    }
    if (prDrawn.length === 0) return ''
    const head = prDrawn
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${px(p.recall).toFixed(2)} ${py(p.precision).toFixed(2)}`)
      .join(' ')
    const firstRecall = prDrawn[0].recall
    const lastRecall = prDrawn[prDrawn.length - 1].recall
    return `${head} L ${px(lastRecall).toFixed(2)} ${py(0).toFixed(2)} L ${px(firstRecall).toFixed(2)} ${py(0).toFixed(2)} Z`
  }, [view, roc, prDrawn])

  const xLabel = view === 'ROC' ? 'false positive rate' : 'recall'
  const yLabel = view === 'ROC' ? 'true positive rate' : 'precision'

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <div className="flex items-center gap-2">
          <span className="text-ink-muted">View</span>
          <div className="inline-flex rounded border border-rule overflow-hidden">
            {(['ROC', 'PR'] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className="px-3 py-1 transition-colors border-l border-rule first:border-l-0"
                style={{
                  background: view === v ? 'var(--accent)' : 'transparent',
                  color: view === v ? 'var(--paper)' : 'var(--ink)',
                }}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
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
            className="w-40"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* plot frame */}
          <rect
            x={PAD_L}
            y={PAD_T}
            width={PLOT}
            height={PLOT}
            fill="none"
            stroke="var(--rule)"
            strokeWidth={1}
          />

          {/* reference line: chance diagonal (ROC) or base-rate (PR) */}
          {view === 'ROC' ? (
            <line
              x1={px(0)}
              y1={py(0)}
              x2={px(1)}
              y2={py(1)}
              stroke="var(--rule)"
              strokeWidth={1}
              strokeDasharray="4,3"
            />
          ) : (
            <line
              x1={px(0)}
              y1={py(posRate)}
              x2={px(1)}
              y2={py(posRate)}
              stroke="var(--rule)"
              strokeWidth={1}
              strokeDasharray="4,3"
            />
          )}

          {/* area under curve */}
          <path d={fillPath} fill="var(--accent)" opacity={0.1} />

          {/* the curve */}
          <path d={curvePath} fill="none" stroke="var(--accent)" strokeWidth={2} />

          {/* operating-point guide lines */}
          <line
            x1={px(markerX)}
            y1={py(markerY)}
            x2={px(markerX)}
            y2={py(0)}
            stroke="var(--ink-muted)"
            strokeWidth={1}
            strokeDasharray="2,2"
            opacity={0.5}
          />
          <line
            x1={px(markerX)}
            y1={py(markerY)}
            x2={px(0)}
            y2={py(markerY)}
            stroke="var(--ink-muted)"
            strokeWidth={1}
            strokeDasharray="2,2"
            opacity={0.5}
          />
          {/* operating-point marker */}
          <circle
            cx={px(markerX)}
            cy={py(markerY)}
            r={5}
            fill="var(--accent)"
            stroke="var(--paper)"
            strokeWidth={1.5}
          />

          {/* axis ticks */}
          {[0, 0.5, 1].map((t) => (
            <g key={`x${t}`}>
              <text
                x={px(t)}
                y={PAD_T + PLOT + 16}
                fontSize={10}
                fill="var(--ink-muted)"
                textAnchor="middle"
                fontFamily="var(--font-mono, monospace)"
              >
                {t.toFixed(1)}
              </text>
            </g>
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
            {xLabel}
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
            {yLabel}
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          {view === 'ROC' ? 'AUC' : 'avg precision'} ={' '}
          <span className="text-ink">
            {(view === 'ROC' ? auc : ap).toFixed(3)}
          </span>
        </div>
        <div>
          {view === 'ROC'
            ? `TPR=${markerY.toFixed(2)}  FPR=${markerX.toFixed(2)}`
            : `prec=${markerY.toFixed(2)}  rec=${markerX.toFixed(2)}`}
        </div>
        <div>positive rate = {(posRate * 100).toFixed(1)}%</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 10.2 — The {view === 'ROC' ? 'ROC' : 'precision-recall'} curve
        for the same scored examples as Figure 10.1. The dot is the operating
        point at the current threshold — slide it and watch the single point
        travel the whole curve. Switch to the imbalanced data and compare the
        two views: the ROC stays flatteringly high while the precision-recall
        curve collapses, because PR never counts the vast pool of true
        negatives.
      </figcaption>
    </figure>
  )
}
