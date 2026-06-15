'use client'

import { useMemo, useState } from 'react'
import {
  generateDataset,
  type DatasetShape,
  type Point,
} from '@/lib/datasets'
import { runKMeans } from '@/lib/kmeans'
import { silhouette } from '@/lib/silhouette'

const CLUSTER_COLORS = [
  '#1d6d5e',
  '#c7522a',
  '#3c5a8c',
  '#a06614',
  '#6b4a8a',
  '#a83263',
  '#5d7a25',
  '#a3322b',
]

const K_VALUES = [2, 3, 4, 5, 6, 7, 8]

export default function KSweepWidget() {
  const [shape, setShape] = useState<DatasetShape>('blobs')
  const [seed, setSeed] = useState(31)
  const [k, setK] = useState(3)

  const points: Point[] = useMemo(
    () => generateDataset(shape, 120, seed),
    [shape, seed],
  )

  const sweep = useMemo(() => {
    return K_VALUES.map((kk) => {
      const history = runKMeans(points, kk, 'kpp', seed + 7)
      const last = history[history.length - 1]
      return {
        k: kk,
        wcss: last.wcss ?? 0,
        sil: silhouette(points, last.assignments, kk),
      }
    })
  }, [points, seed])

  const currentRun = useMemo(() => {
    const h = runKMeans(points, k, 'kpp', seed + 7)
    return h[h.length - 1]
  }, [points, k, seed])

  return (
    <figure className="not-prose my-10">
      <div className="rounded-lg border border-rule bg-paper p-4 sm:p-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
          <label className="flex flex-col gap-1.5">
            <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
              Dataset
            </span>
            <select
              value={shape}
              onChange={(e) => setShape(e.target.value as DatasetShape)}
              className="w-full h-9 rounded-md px-2 border border-rule font-sans text-sm"
              style={{
                background: 'var(--paper)',
                color: 'var(--ink)',
              }}
            >
              <option value="blobs">Three blobs (true K = 3)</option>
              <option value="aniso">Anisotropic (true K = 3)</option>
              <option value="moons">Two moons (true K = 2)</option>
            </select>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="font-sans text-xs uppercase tracking-wider text-ink-muted">
              K = {k}
            </span>
            <input
              type="range"
              min={2}
              max={8}
              step={1}
              value={k}
              onChange={(e) => setK(+e.target.value)}
              style={{ accentColor: 'var(--accent)' }}
            />
          </label>
        </div>

        <svg
          viewBox="0 0 680 280"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full block rounded-md mb-4"
          style={{
            background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))',
            aspectRatio: '680 / 280',
          }}
        >
          {points.map((p, i) => {
            const a = currentRun.assignments[i]
            return (
              <circle
                key={i}
                cx={p[0]}
                cy={p[1] * 0.78}
                r={4}
                fill={CLUSTER_COLORS[a] ?? 'var(--ink-faint)'}
                opacity={0.85}
              />
            )
          })}
          {currentRun.centroids.map((c, kk) => (
            <g
              key={kk}
              transform={`translate(${c[0].toFixed(1)}, ${(c[1] * 0.78).toFixed(1)})`}
            >
              <circle
                r={10}
                fill={CLUSTER_COLORS[kk]}
                stroke="var(--paper)"
                strokeWidth={3}
              />
              <path
                d="M-4,0 L4,0 M0,-4 L0,4"
                stroke="var(--paper)"
                strokeWidth={2}
                strokeLinecap="round"
              />
            </g>
          ))}
        </svg>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SweepChart
            data={sweep.map((s) => ({ k: s.k, value: s.wcss }))}
            currentK={k}
            onPickK={setK}
            title="WCSS by K"
            subtitle="lower is tighter"
            color="var(--accent)"
            higherIsBetter={false}
          />
          <SweepChart
            data={sweep.map((s) => ({ k: s.k, value: s.sil }))}
            currentK={k}
            onPickK={setK}
            title="Silhouette by K"
            subtitle="higher is better separated"
            color="#c7522a"
            higherIsBetter
          />
        </div>

        <div className="mt-3 flex items-center justify-end">
          <button
            onClick={() => setSeed(Math.floor(Math.random() * 100000))}
            className="font-sans text-xs uppercase tracking-wider px-3 py-1.5 rounded-md border border-rule text-ink-muted hover:text-ink hover:border-ink-faint transition-colors"
          >
            New data
          </button>
        </div>
      </div>
      <figcaption className="font-sans text-sm text-ink-muted mt-3 text-center">
        Figure 18.4 — As you sweep K, the WCSS falls monotonically while the
        silhouette peaks. The peak (or the elbow) is your best K.
      </figcaption>
    </figure>
  )
}

function SweepChart({
  data,
  currentK,
  onPickK,
  title,
  subtitle,
  color,
  higherIsBetter,
}: {
  data: { k: number; value: number }[]
  currentK: number
  onPickK: (k: number) => void
  title: string
  subtitle: string
  color: string
  higherIsBetter: boolean
}) {
  const W = 320
  const H = 200
  const padL = 44
  const padR = 14
  const padT = 36
  const padB = 32
  const iw = W - padL - padR
  const ih = H - padT - padB

  const values = data.map((d) => d.value)
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const range = Math.max(1e-9, maxV - minV)

  const xOf = (k: number) =>
    padL + ((k - data[0].k) / (data[data.length - 1].k - data[0].k)) * iw
  const yOf = (v: number) =>
    padT + ih - ((v - minV) / range) * ih * 0.92 - ih * 0.04

  const linePath = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xOf(d.k).toFixed(1)} ${yOf(d.value).toFixed(1)}`)
    .join(' ')

  const bestK = higherIsBetter
    ? data.reduce((a, b) => (b.value > a.value ? b : a)).k
    : null

  return (
    <div
      className="rounded-md p-3"
      style={{ background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))' }}
    >
      <div className="flex items-baseline justify-between mb-2">
        <div>
          <span
            className="font-sans"
            style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}
          >
            {title}
          </span>
          <span
            className="font-sans ml-2"
            style={{ fontSize: 11, color: 'var(--ink-muted)' }}
          >
            {subtitle}
          </span>
        </div>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        xmlns="http://www.w3.org/2000/svg"
        className="w-full block"
        style={{ aspectRatio: `${W} / ${H}` }}
      >
        <line
          x1={padL}
          y1={padT + ih}
          x2={padL + iw}
          y2={padT + ih}
          stroke="var(--ink-faint)"
          strokeWidth={0.5}
        />
        <line
          x1={padL}
          y1={padT}
          x2={padL}
          y2={padT + ih}
          stroke="var(--ink-faint)"
          strokeWidth={0.5}
        />
        <text
          x={padL}
          y={padT - 4}
          fontFamily="var(--font-sans)"
          fontSize={10}
          fill="var(--ink-muted)"
        >
          {higherIsBetter ? maxV.toFixed(2) : Math.round(maxV).toLocaleString()}
        </text>
        <text
          x={padL}
          y={padT + ih + 18}
          fontFamily="var(--font-sans)"
          fontSize={10}
          fill="var(--ink-muted)"
        >
          K
        </text>

        <path
          d={linePath}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          opacity={0.55}
        />
        {data.map((d) => (
          <g key={d.k}>
            <circle
              cx={xOf(d.k)}
              cy={yOf(d.value)}
              r={d.k === currentK ? 6 : 4}
              fill={d.k === currentK ? color : 'var(--paper)'}
              stroke={color}
              strokeWidth={1.5}
              style={{ cursor: 'pointer', transition: 'r 0.2s' }}
              onClick={() => onPickK(d.k)}
            />
            <text
              x={xOf(d.k)}
              y={padT + ih + 18}
              fontFamily="var(--font-sans)"
              fontSize={10}
              fill={d.k === currentK ? 'var(--ink)' : 'var(--ink-muted)'}
              textAnchor="middle"
              style={{ fontWeight: d.k === currentK ? 500 : 400 }}
            >
              {d.k}
            </text>
          </g>
        ))}
        {bestK != null && (
          <text
            x={xOf(bestK)}
            y={yOf(data.find((d) => d.k === bestK)!.value) - 12}
            fontFamily="var(--font-sans)"
            fontSize={10}
            fill={color}
            textAnchor="middle"
            style={{ fontWeight: 500 }}
          >
            peak
          </text>
        )}
      </svg>
    </div>
  )
}
