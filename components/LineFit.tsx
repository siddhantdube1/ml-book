'use client'

import { useMemo, useRef, useState } from 'react'
import { make1DLinear, leastSquaresLine, lineMSE } from '@/lib/foundations'

const W = 480
const H = 360
const PAD = 34
const X0 = 0, X1 = 10, Y0 = 0, Y1 = 11
const XL = 1, XR = 9 // x positions of the two drag handles

export default function LineFit() {
  const svgRef = useRef<SVGSVGElement>(null)
  const data = useMemo(() => make1DLinear(22, 7, 0.82, 1.6, 0.9, 0.5, 9.5), [])
  const [yL, setYL] = useState(5.5)
  const [yR, setYR] = useState(6.5)
  const [drag, setDrag] = useState<'L' | 'R' | null>(null)

  const slope = (yR - yL) / (XR - XL)
  const intercept = yL - slope * XL

  const sx = (x: number) => PAD + ((x - X0) / (X1 - X0)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y - Y0) / (Y1 - Y0)) * (H - 2 * PAD)

  const mse = lineMSE(data, slope, intercept)

  const snap = () => {
    const best = leastSquaresLine(data.map((p) => p.x), data.map((p) => p.y))
    setYL(best.slope * XL + best.intercept)
    setYR(best.slope * XR + best.intercept)
  }

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!drag) return
    const svg = svgRef.current!
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY
    const { y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    const yv = Y0 + ((H - PAD - y) / (H - 2 * PAD)) * (Y1 - Y0)
    const clamped = Math.max(Y0, Math.min(Y1, yv))
    if (drag === 'L') setYL(clamped); else setYR(clamped)
  }

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <button onClick={snap} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">snap to best fit</button>
        <span className="font-sans text-xs text-ink-faint">drag the two handles to tilt and shift the line</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block select-none touch-none"
          onPointerMove={onMove} onPointerUp={(e) => { setDrag(null); e.currentTarget.releasePointerCapture(e.pointerId) }}>
          <line x1={sx(X0)} y1={sy(Y0)} x2={sx(X1)} y2={sy(Y0)} stroke="var(--rule)" strokeWidth={1} />
          <line x1={sx(X0)} y1={sy(Y0)} x2={sx(X0)} y2={sy(Y1)} stroke="var(--rule)" strokeWidth={1} />
          {/* residuals */}
          {data.map((p, i) => <line key={`r${i}`} x1={sx(p.x)} y1={sy(p.y)} x2={sx(p.x)} y2={sy(slope * p.x + intercept)} stroke="#c7522a" strokeWidth={1} opacity={0.55} />)}
          {/* the line */}
          <line x1={sx(X0)} y1={sy(slope * X0 + intercept)} x2={sx(X1)} y2={sy(slope * X1 + intercept)} stroke="var(--accent)" strokeWidth={2.25} />
          {data.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3.4} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.6} />)}
          {/* handles */}
          <circle cx={sx(XL)} cy={sy(yL)} r={8} fill="var(--paper)" stroke="var(--ink)" strokeWidth={2} style={{ cursor: 'ns-resize' }} onPointerDown={(e) => { setDrag('L'); e.currentTarget.setPointerCapture(e.pointerId) }} />
          <circle cx={sx(XR)} cy={sy(yR)} r={8} fill="var(--paper)" stroke="var(--ink)" strokeWidth={2} style={{ cursor: 'ns-resize' }} onPointerDown={(e) => { setDrag('R'); e.currentTarget.setPointerCapture(e.pointerId) }} />
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>ŷ = <span className="text-ink">{slope.toFixed(2)}</span> x + <span className="text-ink">{intercept.toFixed(2)}</span></div>
        <div>mean squared error = <span className="text-accent">{mse.toFixed(3)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 5.1 — Fitting a line. The orange stubs are the <strong>residuals</strong>
        — the gap between each point and the line&rsquo;s prediction for it. The loss,
        the <strong>mean squared error</strong>, is the average of those gaps squared,
        and a good line makes it small. Tilt and shift the line by dragging the
        handles and watch the error rise and fall; then press <em>snap to best fit</em>
        to jump to the one line that minimises it — the least-squares line. Fitting a
        model is exactly this: choosing parameters that make the loss as small as it
        will go.
      </figcaption>
    </figure>
  )
}
