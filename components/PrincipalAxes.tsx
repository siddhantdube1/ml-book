'use client'

import { useMemo, useRef, useState } from 'react'
import { correlatedCloud2D, pca } from '@/lib/pca'

const W = 460
const H = 420
const CX = W / 2
const CY = H / 2
const SCALE = 24
const HANDLE_R = 150

export default function PrincipalAxes() {
  const [angle, setAngle] = useState(1.15) // radians; user drags this
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const data = useMemo(() => correlatedCloud2D(140, 7), [])
  const model = useMemo(() => pca(data), [data])

  // covariance for projected-variance along an arbitrary direction
  const cov = useMemo(() => {
    let cxx = 0, cxy = 0, cyy = 0
    for (const [x, y] of data) { cxx += x * x; cxy += x * y; cyy += y * y }
    const n = data.length
    return { cxx: cxx / n, cxy: cxy / n, cyy: cyy / n }
  }, [data])

  const pc1Angle = Math.atan2(model.components[0][1], model.components[0][0])
  const lam1 = model.eigenvalues[0]
  const lam2 = model.eigenvalues[1]

  const u: [number, number] = [Math.cos(angle), Math.sin(angle)]
  const projVar = u[0] * u[0] * cov.cxx + 2 * u[0] * u[1] * cov.cxy + u[1] * u[1] * cov.cyy
  const captured = projVar / (lam1 + lam2)
  // nearest alignment to PC1 axis (line, so mod π)
  const diff = Math.abs(((angle - pc1Angle + Math.PI / 2 + Math.PI) % Math.PI) - Math.PI / 2)
  const aligned = diff < 0.06

  const sx = (x: number) => CX + x * SCALE
  const sy = (y: number) => CY - y * SCALE

  const setFromPointer = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    setAngle(Math.atan2(-(y - CY), x - CX))
  }

  // direction line endpoints
  const lineDX = Math.cos(angle) * 200
  const lineDY = Math.sin(angle) * 200

  return (
    <figure className="my-10">
      <div className="font-sans text-sm text-ink-muted mb-3">Drag the dashed line to rotate it through the cloud.</div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block touch-none"
          style={{ cursor: dragging ? 'grabbing' : 'default' }}
          onPointerMove={(e) => dragging && setFromPointer(e)}
          onPointerUp={(e) => { setDragging(false); e.currentTarget.releasePointerCapture(e.pointerId) }}
        >
          {/* projection ticks on the candidate line */}
          {data.map(([x, y], i) => {
            const t = x * u[0] + y * u[1]
            return <circle key={`t${i}`} cx={sx(t * u[0])} cy={sy(t * u[1])} r={1.6} fill="var(--ink-faint)" />
          })}
          {/* candidate direction line */}
          <line x1={sx(-lineDX / SCALE)} y1={sy(-lineDY / SCALE)} x2={sx(lineDX / SCALE)} y2={sy(lineDY / SCALE)} stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="5,4" />
          {/* PC axes (revealed) */}
          <line x1={sx(-model.components[0][0] * 6)} y1={sy(-model.components[0][1] * 6)} x2={sx(model.components[0][0] * 6)} y2={sy(model.components[0][1] * 6)} stroke="var(--accent)" strokeWidth={aligned ? 2.5 : 1.5} opacity={aligned ? 1 : 0.5} />
          <line x1={sx(-model.components[1][0] * 2.5)} y1={sy(-model.components[1][1] * 2.5)} x2={sx(model.components[1][0] * 2.5)} y2={sy(model.components[1][1] * 2.5)} stroke="var(--ink)" strokeWidth={1.25} opacity={0.45} />
          {/* data points */}
          {data.map(([x, y], i) => (
            <circle key={i} cx={sx(x)} cy={sy(y)} r={3} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.6} opacity={0.85} />
          ))}
          {/* centroid */}
          <circle cx={CX} cy={CY} r={3.5} fill="var(--ink)" />
          {/* drag handle */}
          <circle
            cx={CX + Math.cos(angle) * HANDLE_R}
            cy={CY - Math.sin(angle) * HANDLE_R}
            r={8}
            fill="var(--paper)"
            stroke="var(--ink)"
            strokeWidth={2}
            style={{ cursor: 'grab' }}
            onPointerDown={(e) => { setDragging(true); e.currentTarget.setPointerCapture(e.pointerId); e.stopPropagation() }}
          />
          {aligned && <text x={sx(model.components[0][0] * 6)} y={sy(model.components[0][1] * 6) - 8} fontSize={11} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">PC1</text>}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>variance along this line = <span className="text-ink">{projVar.toFixed(2)}</span></div>
        <div>captured = <span className={aligned ? 'text-accent' : 'text-ink'}>{(captured * 100).toFixed(1)}%</span>{aligned ? ' ← maximum (PC1)' : ` · max ${((lam1 / (lam1 + lam2)) * 100).toFixed(1)}%`}</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 20.1 — The first principal component. Project the cloud onto the
        dashed line — the small ticks are the shadows — and the variance of those
        shadows swells and shrinks as you rotate it. It is widest along the
        cloud&rsquo;s long axis and narrowest across it. That widest direction,
        the one capturing the most variance of any single direction, is the{' '}
        <span className="text-accent">first principal component</span>; the
        perpendicular one (PC2) gets whatever variance is left. PCA is just this:
        find the directions of greatest spread, in order.
      </figcaption>
    </figure>
  )
}
