'use client'

import { useMemo, useState } from 'react'
import { makeMoons2, type Point } from '@/lib/tree'
import { knnNeighbours } from '@/lib/features'

const W = 420
const H = 380
const PAD = 22
const RANGE_X = 3.2
const RANGE_Y = 2.7
const GX = 56
const GY = 50
const CLASS = ['#3c5a8c', '#c7522a'] as const

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1, g = 90 + (82 - 90) * p1, b = 140 + (42 - 140) * p1
  return `rgba(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)},${(Math.abs(p1 - 0.5) * 2 * 0.5).toFixed(3)})`
}

export default function KNNBoundary() {
  const [k, setK] = useState(1)
  const data = useMemo<Point[]>(() => makeMoons2(120, 1, 0.3), [])

  const sx = (x: number) => PAD + ((x + RANGE_X) / (2 * RANGE_X)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + RANGE_Y) / (2 * RANGE_Y)) * (H - 2 * PAD)

  const heat = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (W - 2 * PAD) / GX
    const ch = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
      const xD = -RANGE_X + ((i + 0.5) / GX) * 2 * RANGE_X
      const yD = RANGE_Y - ((j + 0.5) / GY) * 2 * RANGE_Y
      const nn = knnNeighbours(data, [xD, yD], k)
      let v = 0
      for (const idx of nn) v += data[idx].y
      cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(v / k) })
    }
    return cells
  }, [data, k])

  const looErr = useMemo(() => {
    let wrong = 0
    for (let i = 0; i < data.length; i++) {
      const nn = knnNeighbours(data, data[i].x, k + 1).filter((j) => j !== i).slice(0, k)
      let v = 0
      for (const j of nn) v += data[j].y
      if ((v * 2 >= k ? 1 : 0) !== data[i].y) wrong++
    }
    return wrong / data.length
  }, [data, k])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">k = <span className="font-mono">{k}</span></span><input type="range" min={1} max={45} step={1} value={k} onChange={(e) => setK(parseInt(e.target.value))} className="w-52" /></label>
        <span className="font-mono text-xs" style={{ color: k <= 2 ? '#c7522a' : k >= 35 ? '#c7522a' : 'var(--accent)' }}>{k <= 2 ? 'jagged (high variance)' : k >= 35 ? 'over-smoothed (high bias)' : 'balanced'}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heat.map((c, i) => <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />)}
          {data.map((p, i) => <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />)}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>k = <span className="text-ink">{k}</span> neighbours vote</div>
        <div>leave-one-out error = <span className="text-accent">{(looErr * 100).toFixed(1)}%</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 4.1 — k tunes the bias–variance trade-off. With{' '}
        <span className="font-mono">k = 1</span> every point gets its own little
        island and the boundary is jagged — it follows the noise (high variance,
        overfitting). Crank k up and the vote averages over a wide neighbourhood, so
        the boundary smooths out — until, with k too large, it blurs across the real
        gap between the classes (high bias, underfitting). Somewhere in the middle is
        the sweet spot, found exactly as in Chapter 2: by held-out error, not by
        eye.
      </figcaption>
    </figure>
  )
}
