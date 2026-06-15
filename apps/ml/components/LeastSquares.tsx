'use client'

import { useMemo, useState } from 'react'
import { make1DLinear, leastSquaresLine } from '@/lib/foundations'

const W = 480
const H = 380
const SCALE = 26 // px per unit, SAME for x and y so the squares look square
const ORX = 40
const ORY = H - 30

export default function LeastSquares() {
  const data = useMemo(() => make1DLinear(12, 4, 0.8, 1.8, 0.8, 1, 9), [])
  const mx = data.reduce((a, p) => a + p.x, 0) / data.length
  const my = data.reduce((a, p) => a + p.y, 0) / data.length
  const best = useMemo(() => leastSquaresLine(data.map((p) => p.x), data.map((p) => p.y)), [data])

  const [slope, setSlope] = useState(0.2)
  const line = (x: number) => slope * (x - mx) + my // pivots through the centroid

  const sx = (x: number) => ORX + x * SCALE
  const sy = (y: number) => ORY - y * SCALE

  const totalArea = data.reduce((a, p) => a + (p.y - line(p.x)) ** 2, 0)
  const atBest = Math.abs(slope - best.slope) < 0.03

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">slope = <span className="font-mono">{slope.toFixed(2)}</span></span><input type="range" min={-0.4} max={1.6} step={0.02} value={slope} onChange={(e) => setSlope(parseFloat(e.target.value))} className="w-52" /></label>
        <span className="font-mono text-xs" style={{ color: atBest ? 'var(--accent)' : 'var(--ink-muted)' }}>{atBest ? 'least squares!' : `best slope = ${best.slope.toFixed(2)}`}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={ORX} y1={ORY} x2={W - 14} y2={ORY} stroke="var(--rule)" strokeWidth={1} />
          <line x1={ORX} y1={ORY} x2={ORX} y2={20} stroke="var(--rule)" strokeWidth={1} />
          {/* residual squares: side = |residual|, so area = residual^2 */}
          {data.map((p, i) => {
            const yhat = line(p.x)
            const r = p.y - yhat
            const side = Math.abs(r) * SCALE
            const top = sy(Math.max(p.y, yhat))
            return <rect key={`sq${i}`} x={sx(p.x)} y={top} width={side} height={side} fill="#c7522a" opacity={0.18} stroke="#c7522a" strokeWidth={0.5} strokeOpacity={0.4} />
          })}
          {/* the line */}
          <line x1={sx(0)} y1={sy(line(0))} x2={sx(10)} y2={sy(line(10))} stroke="var(--accent)" strokeWidth={2.25} />
          {data.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3.6} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.7} />)}
        </svg>
      </div>

      <div className="font-mono text-xs text-ink-muted mt-3">total area of the squares (the loss) = <span style={{ color: atBest ? 'var(--accent)' : 'var(--ink)' }}>{totalArea.toFixed(2)}</span></div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 5.2 — Why &ldquo;least squares&rdquo;. Each residual is turned into an
        actual square whose side is the size of the miss — so its area is the residual
        <em> squared</em>, and the total orange area is exactly the loss. Tilt the line
        and the squares swell and shrink; the best-fit line is, quite literally, the one
        that makes the <strong>total area smallest</strong>. That is all
        &ldquo;least squares&rdquo; means — and squaring (rather than, say, absolute
        value) is what makes the minimum solvable in one clean formula.
      </figcaption>
    </figure>
  )
}
