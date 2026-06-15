'use client'

import { useMemo, useState } from 'react'
import { makeScaleData, fitScaler, knnNeighbours } from '@/lib/features'
import type { Point } from '@/lib/tree'

const W = 440
const H = 400
const PAD = 30
const GX = 50
const GY = 46
const K = 7

const CLASS = ['#3c5a8c', '#c7522a'] as const
const QUERY: [number, number] = [500, 0.5] // raw x1 in [0,1000], x2 in [0,1]

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const alpha = Math.abs(p1 - 0.5) * 2 * 0.42
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

export default function FeatureScaling() {
  const [standardise, setStandardise] = useState(false)

  const data = useMemo<Point[]>(() => makeScaleData(160, 7), [])
  const scaler = useMemo(() => fitScaler(data.map((p) => p.x)), [data])
  const active = standardise ? scaler : undefined

  // plot in normalised coords: x1/1000 -> [0,1], x2 -> [0,1]
  const sx = (nx: number) => PAD + nx * (W - 2 * PAD)
  const sy = (ny: number) => H - PAD - ny * (H - 2 * PAD)

  const knnProb = (x1: number, x2: number): number => {
    const nn = knnNeighbours(data, [x1, x2], K, active)
    let v = 0
    for (const i of nn) v += data[i].y
    return v / K
  }

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (W - 2 * PAD) / GX
    const ch = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const nx = (i + 0.5) / GX
        const ny = 1 - (j + 0.5) / GY
        cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(knnProb(nx * 1000, ny)) })
      }
    }
    return cells
  }, [active])

  const nbrs = useMemo(() => knnNeighbours(data, QUERY, K, active), [active])

  const acc = useMemo(() => {
    let ok = 0
    for (const p of data) {
      const nn = knnNeighbours(data, p.x, K + 1, active).filter((i) => data[i] !== p).slice(0, K)
      let v = 0
      for (const i of nn) v += data[i].y
      if ((v * 2 >= K ? 1 : 0) === p.y) ok++
    }
    return ok / data.length
  }, [active])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={standardise} onChange={(e) => setStandardise(e.target.checked)} />
          <span className="text-ink-muted">standardise features</span>
        </label>
        <span className="font-mono text-xs text-ink-faint">k = {K} · query ◇ at (x₁ = 500, x₂ = 0.5)</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          {/* lines from query to its neighbours */}
          {nbrs.map((i) => (
            <line key={`l${i}`} x1={sx(QUERY[0] / 1000)} y1={sy(QUERY[1])} x2={sx(data[i].x[0] / 1000)} y2={sy(data[i].x[1])} stroke="var(--ink)" strokeWidth={0.6} opacity={0.5} />
          ))}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0] / 1000)} cy={sy(p.x[1])} r={nbrs.includes(i) ? 5 : 3} fill={CLASS[p.y]} stroke={nbrs.includes(i) ? 'var(--ink)' : 'var(--paper)'} strokeWidth={nbrs.includes(i) ? 1.5 : 0.7} />
          ))}
          {/* query */}
          <rect x={sx(QUERY[0] / 1000) - 5} y={sy(QUERY[1]) - 5} width={10} height={10} fill="var(--paper)" stroke="var(--ink)" strokeWidth={2} transform={`rotate(45 ${sx(QUERY[0] / 1000)} ${sy(QUERY[1])})`} />
          {/* axis labels */}
          <text x={W / 2} y={H - 8} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">x₁ (range 0–1000)</text>
          <text x={12} y={H / 2} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic" transform={`rotate(-90 12 ${H / 2})`}>x₂ (range 0–1)</text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>distance metric = <span className="text-ink">{standardise ? 'standardised' : 'raw'}</span></div>
        <div>leave-one-out kNN accuracy = <span className="text-accent">{(acc * 100).toFixed(1)}%</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 17.1 — Why scale matters. The class depends on{' '}
        <em>both</em> features, but x₁ runs to 1000 while x₂ never leaves
        [0, 1], so in raw
        units the distance between two points is almost entirely x₁ — and the
        k-nearest-neighbour boundary comes out nearly vertical, deaf to x₂. The
        ringed points are the query&rsquo;s seven nearest neighbours; tick{' '}
        <span className="font-mono">standardise</span> and they jump to an
        entirely different set as x₂ is finally heard, the boundary turns
        diagonal, and the accuracy climbs. Same data, same model — only the
        units changed.
      </figcaption>
    </figure>
  )
}
