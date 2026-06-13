'use client'

import { useRef, useState } from 'react'

const W = 680
const H = 340

// left: 1-D function f(x) = 0.35 x^3 - 0.7 x  on x in [-2.2, 2.2]
const f1 = (x: number) => 0.35 * x ** 3 - 0.7 * x
const df1 = (x: number) => 1.05 * x ** 2 - 0.7
const LX = 20, LW = 300, XR = 2.2, YR = 1.8

// right: 2-D bowl f(x,y) = 0.5(x^2 + 3 y^2); grad = (x, 3y)
const grad2 = (x: number, y: number): [number, number] => [x, 3 * y]
const RX = 360, RW = 300, R2 = 2.4
const RCX = RX + RW / 2, RCY = H / 2

export default function SlopeAndGradient() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [x1, setX1] = useState(-1.1)
  const [p2, setP2] = useState<[number, number]>([1.5, 0.9])
  const [drag, setDrag] = useState<'a' | 'b' | null>(null)

  const lsx = (x: number) => LX + 10 + ((x + XR) / (2 * XR)) * (LW - 20)
  const lsy = (y: number) => H - 24 - ((y + YR) / (2 * YR)) * (H - 54)
  const rsx = (x: number) => RCX + x * (RW / 2 / R2 - 6)
  const rsy = (y: number) => RCY - y * ((H - 54) / 2 / R2)

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current!
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    if (drag === 'a') setX1(Math.max(-XR, Math.min(XR, ((x - LX - 10) / (LW - 20)) * 2 * XR - XR)))
    if (drag === 'b') setP2([Math.max(-R2, Math.min(R2, (x - RCX) / (RW / 2 / R2 - 6))), Math.max(-R2, Math.min(R2, (RCY - y) / ((H - 54) / 2 / R2)))])
  }

  // tangent line at x1
  const m = df1(x1), y0 = f1(x1)
  const tx0 = x1 - 0.9, tx1 = x1 + 0.9
  // gradient arrow
  const g = grad2(p2[0], p2[1])
  const gn = Math.hypot(g[0], g[1]) || 1
  const gl = 0.9
  const gtip: [number, number] = [p2[0] + (g[0] / gn) * gl, p2[1] + (g[1] / gn) * gl]
  const ang = Math.atan2(rsy(gtip[1]) - rsy(p2[1]), rsx(gtip[0]) - rsx(p2[0]))

  return (
    <figure className="my-10">
      <div className="font-sans text-sm text-ink-muted mb-3">Drag the point on each panel.</div>
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block select-none touch-none"
          onPointerMove={onMove} onPointerUp={(e) => { setDrag(null); e.currentTarget.releasePointerCapture(e.pointerId) }}>
          <text x={LX + LW / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">derivative = slope of the tangent</text>
          <text x={RCX} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">gradient = steepest-uphill direction</text>
          <line x1={340} y1={28} x2={340} y2={H - 14} stroke="var(--rule)" strokeWidth={1} />

          {/* left: curve + tangent */}
          <line x1={lsx(-XR)} y1={lsy(0)} x2={lsx(XR)} y2={lsy(0)} stroke="var(--rule)" strokeWidth={0.6} />
          <path d={Array.from({ length: 121 }, (_, k) => { const x = -XR + (k / 120) * 2 * XR; return `${k === 0 ? 'M' : 'L'} ${lsx(x).toFixed(1)} ${lsy(f1(x)).toFixed(1)}` }).join(' ')} fill="none" stroke="var(--ink)" strokeWidth={2} />
          <line x1={lsx(tx0)} y1={lsy(y0 + m * (tx0 - x1))} x2={lsx(tx1)} y2={lsy(y0 + m * (tx1 - x1))} stroke="var(--accent)" strokeWidth={2} />
          <circle cx={lsx(x1)} cy={lsy(y0)} r={6} fill="var(--paper)" stroke="var(--accent)" strokeWidth={2} style={{ cursor: 'grab' }} onPointerDown={(e) => { setDrag('a'); e.currentTarget.setPointerCapture(e.pointerId) }} />
          <text x={LX + 12} y={H - 8} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-mono, monospace)">slope f′(x) = {m.toFixed(2)}</text>

          {/* right: contour ellipses + gradient */}
          {[0.4, 1, 1.9, 3, 4.4].map((c, i) => (
            <ellipse key={i} cx={RCX} cy={RCY} rx={Math.sqrt(2 * c) * (RW / 2 / R2 - 6)} ry={Math.sqrt(2 * c / 3) * ((H - 54) / 2 / R2)} fill="none" stroke="var(--rule)" strokeWidth={1} />
          ))}
          <line x1={rsx(p2[0])} y1={rsy(p2[1])} x2={rsx(gtip[0])} y2={rsy(gtip[1])} stroke="var(--accent)" strokeWidth={2.5} />
          <polygon points={`${rsx(gtip[0])},${rsy(gtip[1])} ${rsx(gtip[0]) - 9 * Math.cos(ang - 0.4)},${rsy(gtip[1]) - 9 * Math.sin(ang - 0.4)} ${rsx(gtip[0]) - 9 * Math.cos(ang + 0.4)},${rsy(gtip[1]) - 9 * Math.sin(ang + 0.4)}`} fill="var(--accent)" />
          <circle cx={rsx(p2[0])} cy={rsy(p2[1])} r={6} fill="var(--paper)" stroke="var(--ink)" strokeWidth={2} style={{ cursor: 'grab' }} onPointerDown={(e) => { setDrag('b'); e.currentTarget.setPointerCapture(e.pointerId) }} />
          <text x={RX + 12} y={H - 8} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-mono, monospace)">∇f = ({g[0].toFixed(1)}, {g[1].toFixed(1)})</text>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 3.2 — Slopes and gradients. In one dimension (left), the{' '}
        <strong>derivative</strong> $f'(x)$ is the slope of the tangent line — how
        steeply the function rises, and which way. In two dimensions (right), the{' '}
        <strong>gradient</strong> $\nabla f$ collects the slope in each direction
        into an arrow that points straight <em>uphill</em>, perpendicular to the
        contour lines and longest where the surface is steepest. Chapter 6's
        gradient descent does just one thing: compute this arrow and step the
        opposite way, downhill, again and again.
      </figcaption>
    </figure>
  )
}
