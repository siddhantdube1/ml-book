'use client'

import { useRef, useState } from 'react'

const W = 400
const H = 360
const CX = W / 2
const CY = H / 2
const S = 52 // px per unit

type V = [number, number]

export default function DotProduct() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [a, setA] = useState<V>([2, -1])
  const [b, setB] = useState<V>([1.4, 1.7])
  const [drag, setDrag] = useState<'a' | 'b' | null>(null)

  const sx = (x: number) => CX + x * S
  const sy = (y: number) => CY - y * S

  const dot = a[0] * b[0] + a[1] * b[1]
  const na = Math.hypot(a[0], a[1])
  const nb = Math.hypot(b[0], b[1])
  const cos = na && nb ? dot / (na * nb) : 0
  // projection of b onto a
  const k = na ? dot / (na * na) : 0
  const proj: V = [k * a[0], k * a[1]]

  const fromPointer = (e: React.PointerEvent<SVGSVGElement>): V => {
    const svg = svgRef.current!
    const pt = svg.createSVGPoint()
    pt.x = e.clientX; pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    return [(x - CX) / S, (CY - y) / S]
  }

  const arrow = (v: V, col: string, label: string, w = 2.5) => {
    const ang = Math.atan2(sy(v[1]) - CY, sx(v[0]) - CX)
    const tip: V = [sx(v[0]), sy(v[1])]
    const h = 9
    return (
      <g stroke={col} fill={col}>
        <line x1={CX} y1={CY} x2={tip[0]} y2={tip[1]} strokeWidth={w} />
        <polygon points={`${tip[0]},${tip[1]} ${tip[0] - h * Math.cos(ang - 0.4)},${tip[1] - h * Math.sin(ang - 0.4)} ${tip[0] - h * Math.cos(ang + 0.4)},${tip[1] - h * Math.sin(ang + 0.4)}`} stroke="none" />
        <text x={tip[0] + 8 * Math.sign(v[0] || 1)} y={tip[1] - 6} fontSize={13} fontFamily="var(--font-mono, monospace)" stroke="none">{label}</text>
      </g>
    )
  }

  const dotCol = dot > 0.05 ? 'var(--accent)' : dot < -0.05 ? '#c7522a' : 'var(--ink-muted)'

  return (
    <figure className="my-10">
      <div className="font-sans text-sm text-ink-muted mb-3">Drag the tips of the two vectors.</div>
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block select-none touch-none"
          onPointerMove={(e) => { if (drag === 'a') setA(fromPointer(e)); if (drag === 'b') setB(fromPointer(e)) }}
          onPointerUp={(e) => { setDrag(null); e.currentTarget.releasePointerCapture(e.pointerId) }}
        >
          {/* axes */}
          <line x1={0} y1={CY} x2={W} y2={CY} stroke="var(--rule)" strokeWidth={0.75} />
          <line x1={CX} y1={0} x2={CX} y2={H} stroke="var(--rule)" strokeWidth={0.75} />
          {/* projection of b onto a */}
          <line x1={sx(proj[0])} y1={sy(proj[1])} x2={sx(b[0])} y2={sy(b[1])} stroke="var(--ink-muted)" strokeWidth={1} strokeDasharray="3,3" />
          <line x1={CX} y1={CY} x2={sx(proj[0])} y2={sy(proj[1])} stroke="var(--ink)" strokeWidth={4} opacity={0.25} />
          {arrow(a, 'var(--accent)', 'a')}
          {arrow(b, '#3c5a8c', 'b')}
          {/* drag handles */}
          <circle cx={sx(a[0])} cy={sy(a[1])} r={9} fill="transparent" style={{ cursor: 'grab' }} onPointerDown={(e) => { setDrag('a'); e.currentTarget.setPointerCapture(e.pointerId) }} />
          <circle cx={sx(b[0])} cy={sy(b[1])} r={9} fill="transparent" style={{ cursor: 'grab' }} onPointerDown={(e) => { setDrag('b'); e.currentTarget.setPointerCapture(e.pointerId) }} />
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>a · b = <span style={{ color: dotCol }}>{dot.toFixed(2)}</span> = |a||b|cos θ</div>
        <div>cos θ = <span className="text-ink">{cos.toFixed(2)}</span> {dot > 0.05 ? '(aligned)' : dot < -0.05 ? '(opposed)' : '(perpendicular)'}</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 3.1 — The dot product measures alignment. It is the length of one
        vector times how much of the other points the same way — the grey shadow is
        the projection of <span style={{ color: '#3c5a8c' }}>b</span> onto{' '}
        <span style={{ color: 'var(--accent)' }}>a</span>. It is large and positive
        when the vectors point together, exactly zero when they are perpendicular,
        and negative when they oppose. Every linear model in this book scores an
        input by exactly this operation — the weighted sum $w \cdot x$ is a dot
        product, asking how well the input aligns with the weights.
      </figcaption>
    </figure>
  )
}
