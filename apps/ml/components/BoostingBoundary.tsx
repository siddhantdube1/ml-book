'use client'

import { useMemo, useState } from 'react'
import { makeMoons2, type Point } from '@/lib/tree'
import { trainGBClassifier, gbClassProb } from '@/lib/boosting'

const PW = 420
const PH = 360
const PAD = 24
const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.6
const Y_MAX = 2.6
const GX = 56
const GY = 48
const MAX_B = 80

const CLASS = ['#3c5a8c', '#c7522a'] as const

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const certainty = Math.abs(p1 - 0.5) * 2
  const alpha = certainty * 0.5
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

export default function BoostingBoundary() {
  const [b, setB] = useState(40)
  const [depth, setDepth] = useState(1)

  const data = useMemo<Point[]>(() => makeMoons2(160, 1, 0.28), [])

  const model = useMemo(
    () => trainGBClassifier(data, { numTrees: MAX_B, learningRate: 0.3, maxDepth: depth, minLeaf: 2 }),
    [data, depth],
  )

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (PW - 2 * PAD)
  const sy = (y: number) => PH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (PH - 2 * PAD)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (PW - 2 * PAD) / GX
    const ch = (PH - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yD = Y_MAX - ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(gbClassProb(model, [xD, yD], b)) })
      }
    }
    return cells
  }, [model, b])

  const trainAcc = useMemo(() => {
    let ok = 0
    for (const p of data) if ((gbClassProb(model, p.x, b) >= 0.5 ? 1 : 0) === p.y) ok++
    return ok / data.length
  }, [data, model, b])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Trees = <span className="font-mono">{b}</span></span>
          <input type="range" min={1} max={MAX_B} step={1} value={b} onChange={(e) => setB(parseInt(e.target.value))} className="w-44" />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Weak-learner depth = <span className="font-mono">{depth}</span></span>
          <input type="range" min={1} max={3} step={1} value={depth} onChange={(e) => setDepth(parseInt(e.target.value))} className="w-24" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.7} />
          ))}
          <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            boosted boundary after {b} {b === 1 ? 'tree' : 'trees'}
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>trees = <span className="text-ink">{b}</span> · depth {depth}</div>
        <div>training accuracy = <span className="text-accent">{(trainAcc * 100).toFixed(1)}%</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 14.4 — Boosting a classifier. With weak-learner depth 1, the
        first tree is a single crude cut, and the boundary is nearly useless.
        Add trees and watch it build: each new stump bends the boundary a
        little more around the two arcs, the probabilities hardening from the
        well-separated regions inward. Where Chapter 13's forest averaged
        hundreds of independent trees into a smooth boundary all at once,
        boosting assembles its boundary one deliberate correction at a time.
      </figcaption>
    </figure>
  )
}
