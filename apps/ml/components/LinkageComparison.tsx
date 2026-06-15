'use client'

import { useMemo, useState } from 'react'
import { createRng, gauss } from '@/lib/rng'
import type { Point } from '@/lib/datasets'
import { agglomerative, cutTreeK, type Linkage } from '@/lib/clustering'

const W = 680
const H = 380
const PALETTE = ['#1d6d5e', '#c7522a', '#3c5a8c', '#a06614']

const LINKS: { key: Linkage; label: string }[] = [
  { key: 'single', label: 'single (nearest pair)' },
  { key: 'complete', label: 'complete (farthest pair)' },
  { key: 'average', label: 'average' },
  { key: 'ward', label: 'ward (min variance)' },
]

// two blobs joined by a thin bridge — the chaining stress test
function bridgedData(): Point[] {
  const rng = createRng(3)
  const out: Point[] = []
  for (let i = 0; i < 34; i++) out.push([gauss(rng, 200, 26), gauss(rng, 200, 30)])
  for (let i = 0; i < 34; i++) out.push([gauss(rng, 480, 26), gauss(rng, 200, 30)])
  for (let i = 0; i < 12; i++) out.push([200 + (480 - 200) * (i / 11), 200 + gauss(rng, 0, 5)])
  return out
}

export default function LinkageComparison() {
  const [k, setK] = useState(2)
  const points = useMemo(() => bridgedData(), [])

  const panels = useMemo(
    () =>
      LINKS.map((lk) => ({
        ...lk,
        labels: cutTreeK(agglomerative(points, lk.key), points.length, k),
      })),
    [points, k],
  )

  const bounds = useMemo(() => {
    const xs = points.map((p) => p[0])
    const ys = points.map((p) => p[1])
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
  }, [points])

  const PW = 332
  const PH = 168
  const panelXY = [
    [6, 28],
    [342, 28],
    [6, 204],
    [342, 204],
  ]
  const px = (x: number, ox: number) => ox + 14 + ((x - bounds.x0) / (bounds.x1 - bounds.x0)) * (PW - 28)
  const py = (y: number, oy: number) => oy + 24 + ((y - bounds.y0) / (bounds.y1 - bounds.y0)) * (PH - 38)

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Cut into <span className="font-mono">{k}</span> clusters</span>
          <input type="range" min={2} max={4} step={1} value={k} onChange={(e) => setK(parseInt(e.target.value))} className="w-32" />
        </label>
        <span className="font-mono text-xs text-ink-faint">same data, four definitions of &ldquo;closest&rdquo;</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {panels.map((panel, pi) => {
            const [ox, oy] = panelXY[pi]
            const nClusters = new Set(panel.labels).size
            return (
              <g key={panel.key}>
                <rect x={ox} y={oy} width={PW} height={PH} fill="none" stroke="var(--rule)" strokeWidth={0.75} rx={4} />
                <text x={ox + 10} y={oy + 16} fontSize={10.5} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">{panel.label}</text>
                {points.map((p, i) => (
                  <circle key={i} cx={px(p[0], ox)} cy={py(p[1], oy)} r={3} fill={PALETTE[panel.labels[i] % PALETTE.length]} stroke="var(--paper)" strokeWidth={0.5} />
                ))}
                <text x={ox + PW - 10} y={oy + 16} fontSize={9} fill="var(--ink-faint)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{nClusters} found</text>
              </g>
            )
          })}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 19.2 — Linkage is the whole game. The two blobs are joined by a
        thin bridge of points. <em>Single</em> linkage measures cluster distance
        by the nearest pair, so it happily hops across the bridge and chains both
        blobs into one cluster — the classic failure. <em>Complete</em> and{' '}
        <em>Ward</em> measure by the farthest pair or the variance increase, refuse
        the bridge, and keep the two blobs apart. Same points, same algorithm —
        only the definition of &ldquo;closest&rdquo; changed.
      </figcaption>
    </figure>
  )
}
