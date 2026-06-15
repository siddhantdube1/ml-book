'use client'

import { useMemo, useState } from 'react'
import { generateDataset, type Point } from '@/lib/datasets'
import { dbscan } from '@/lib/clustering'

const W = 660
const H = 380
const PAD = 28
const PALETTE = ['#1d6d5e', '#c7522a', '#3c5a8c', '#a06614', '#6b4a8a', '#a83263', '#5d7a25']

export default function DBSCANPlayback() {
  const [eps, setEps] = useState(22)
  const [minPts, setMinPts] = useState(4)

  const points = useMemo<Point[]>(() => generateDataset('moons', 160, 5), [])
  const result = useMemo(() => dbscan(points, eps, minPts), [points, eps, minPts])

  const bounds = useMemo(() => {
    const xs = points.map((p) => p[0])
    const ys = points.map((p) => p[1])
    const mx = 0.08
    const rx = Math.max(...xs) - Math.min(...xs)
    const ry = Math.max(...ys) - Math.min(...ys)
    return { x0: Math.min(...xs) - mx * rx, x1: Math.max(...xs) + mx * rx, y0: Math.min(...ys) - mx * ry, y1: Math.max(...ys) + mx * ry }
  }, [points])

  // uniform scale so eps-discs stay circular
  const scale = Math.min((W - 2 * PAD) / (bounds.x1 - bounds.x0), (H - 2 * PAD) / (bounds.y1 - bounds.y0))
  const px = (x: number) => PAD + (x - bounds.x0) * scale
  const py = (y: number) => PAD + (y - bounds.y0) * scale

  const colour = (i: number) => (result.labels[i] < 0 ? 'var(--ink-faint)' : PALETTE[result.labels[i] % PALETTE.length])
  const noiseCount = result.labels.filter((l) => l < 0).length
  const coreCount = result.roles.filter((r) => r === 'core').length

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">ε = <span className="font-mono">{eps}</span></span>
          <input type="range" min={6} max={50} step={1} value={eps} onChange={(e) => setEps(parseInt(e.target.value))} className="w-44" />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">minPts = <span className="font-mono">{minPts}</span></span>
          <input type="range" min={3} max={9} step={1} value={minPts} onChange={(e) => setMinPts(parseInt(e.target.value))} className="w-28" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* faint eps-discs on core points -> dense regions glow */}
          {points.map((p, i) =>
            result.roles[i] === 'core' ? (
              <circle key={`d${i}`} cx={px(p[0])} cy={py(p[1])} r={eps * scale} fill={colour(i)} opacity={0.045} />
            ) : null,
          )}
          {/* points by role */}
          {points.map((p, i) => {
            const role = result.roles[i]
            if (role === 'noise') {
              const x = px(p[0])
              const y = py(p[1])
              return (
                <g key={i} stroke="var(--ink-faint)" strokeWidth={1.1}>
                  <line x1={x - 3} y1={y - 3} x2={x + 3} y2={y + 3} />
                  <line x1={x - 3} y1={y + 3} x2={x + 3} y2={y - 3} />
                </g>
              )
            }
            return (
              <circle
                key={i}
                cx={px(p[0])}
                cy={py(p[1])}
                r={role === 'core' ? 4 : 3.5}
                fill={role === 'core' ? colour(i) : 'var(--paper)'}
                stroke={colour(i)}
                strokeWidth={role === 'core' ? 0.7 : 1.6}
              />
            )
          })}
          {/* legend */}
          <g transform={`translate(${W - 150}, 20)`} fontFamily="var(--font-sans, sans-serif)" fontSize={10}>
            <circle cx={6} cy={0} r={4} fill="var(--ink-muted)" /><text x={16} y={3.5} fill="var(--ink-muted)">core (dense)</text>
            <circle cx={6} cy={16} r={3.5} fill="var(--paper)" stroke="var(--ink-muted)" strokeWidth={1.6} /><text x={16} y={19.5} fill="var(--ink-muted)">border</text>
            <g stroke="var(--ink-faint)" strokeWidth={1.1}><line x1={3} y1={29} x2={9} y2={35} /><line x1={3} y1={35} x2={9} y2={29} /></g><text x={16} y={35.5} fill="var(--ink-muted)">noise</text>
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>clusters = <span className="text-accent">{result.numClusters}</span></div>
        <div>core points = <span className="text-ink">{coreCount}</span></div>
        <div>noise = <span className="text-ink">{noiseCount}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 19.3 — DBSCAN by density. A point is <em>core</em> (filled) if at
        least minPts points lie within ε — the faint discs pile up wherever the
        plane is crowded. <em>Border</em> points (ringed) sit at a cluster&rsquo;s
        edge, within ε of a core but not dense themselves; <em>noise</em> points
        (grey ×) are too isolated to join anyone and are left out entirely.
        Shrink ε and clusters shatter into noise; grow it and they swallow the
        gaps and fuse. There is no k — DBSCAN counts the clusters itself.
      </figcaption>
    </figure>
  )
}
