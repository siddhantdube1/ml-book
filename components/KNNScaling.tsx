'use client'

import { useMemo, useState } from 'react'
import { makeScaleData, fitScaler, knnNeighbours } from '@/lib/features'
import type { Point } from '@/lib/tree'

const W = 420
const H = 380
const PAD = 28
const GX = 50
const GY = 46
const K = 7
const CLASS = ['#3c5a8c', '#c7522a'] as const

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1, g = 90 + (82 - 90) * p1, b = 140 + (42 - 140) * p1
  return `rgba(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)},${(Math.abs(p1 - 0.5) * 2 * 0.42).toFixed(3)})`
}

export default function KNNScaling() {
  const [standardise, setStandardise] = useState(false)
  const data = useMemo<Point[]>(() => makeScaleData(150, 7), [])
  const scaler = useMemo(() => fitScaler(data.map((p) => p.x)), [data])
  const active = standardise ? scaler : undefined

  const sx = (nx: number) => PAD + nx * (W - 2 * PAD)
  const sy = (ny: number) => H - PAD - ny * (H - 2 * PAD)

  const heat = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (W - 2 * PAD) / GX
    const ch = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
      const nx = (i + 0.5) / GX, ny = 1 - (j + 0.5) / GY
      const nn = knnNeighbours(data, [nx * 1000, ny], K, active)
      let v = 0
      for (const idx of nn) v += data[idx].y
      cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(v / K) })
    }
    return cells
  }, [data, active])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" checked={standardise} onChange={(e) => setStandardise(e.target.checked)} /><span className="text-ink-muted">standardise the features</span></label>
        <span className="font-mono text-xs text-ink-faint">k = {K} · x₁ ∈ [0, 1000], x₂ ∈ [0, 1]</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heat.map((c, i) => <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />)}
          {data.map((p, i) => <circle key={i} cx={sx(p.x[0] / 1000)} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />)}
          <text x={W / 2} y={H - 8} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">x₁ (range 0–1000)</text>
          <text x={12} y={H / 2} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic" transform={`rotate(-90 12 ${H / 2})`}>x₂ (range 0–1)</text>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 4.3 — Distance depends on units. The class depends on <em>both</em>
        features, but x₁ runs to 1000 while x₂ never leaves [0, 1], so in raw units
        the distance between two points is almost entirely x₁ — and kNN&rsquo;s
        boundary comes out nearly vertical, deaf to x₂. Tick{' '}
        <span className="font-mono">standardise</span> to put both features on the
        same footing and the boundary turns diagonal, finally using both. Any
        distance-based model needs this; Chapter 17 makes a craft of it.
      </figcaption>
    </figure>
  )
}
