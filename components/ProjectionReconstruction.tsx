'use client'

import { useMemo, useState } from 'react'
import { correlatedCloud2D, pca, reconstruct } from '@/lib/pca'

const W = 460
const H = 400
const CX = W / 2
const CY = H / 2 + 10
const SCALE = 24

export default function ProjectionReconstruction() {
  const [k, setK] = useState(1)
  const data = useMemo(() => correlatedCloud2D(120, 7), [])
  const model = useMemo(() => pca(data), [data])
  const recon = useMemo(() => reconstruct(model, data, k), [model, data, k])

  const sx = (x: number) => CX + x * SCALE
  const sy = (y: number) => CY - y * SCALE

  const lam1 = model.eigenvalues[0]
  const lam2 = model.eigenvalues[1]
  const retained = k === 1 ? lam1 / (lam1 + lam2) : 1
  const error = k === 1 ? lam2 : 0
  const p1 = model.components[0]

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <div className="flex rounded border border-rule overflow-hidden">
          {[1, 2].map((kk) => (
            <button key={kk} onClick={() => setK(kk)} className="px-3 py-1 transition-colors" style={{ background: k === kk ? 'var(--accent)' : 'transparent', color: k === kk ? 'var(--paper)' : 'var(--ink-muted)' }}>
              keep {kk} component{kk > 1 ? 's' : ''}
            </button>
          ))}
        </div>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* PC1 line */}
          <line x1={sx(-p1[0] * 7)} y1={sy(-p1[1] * 7)} x2={sx(p1[0] * 7)} y2={sy(p1[1] * 7)} stroke="var(--accent)" strokeWidth={1.5} opacity={0.7} />
          <text x={sx(p1[0] * 7)} y={sy(p1[1] * 7) - 6} fontSize={10} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">PC1</text>

          {/* residual segments (only when compressing to 1-D) */}
          {k === 1 && data.map(([x, y], i) => (
            <line key={`r${i}`} x1={sx(x)} y1={sy(y)} x2={sx(recon[i][0])} y2={sy(recon[i][1])} stroke="#c7522a" strokeWidth={0.8} opacity={0.45} />
          ))}
          {/* original points */}
          {data.map(([x, y], i) => (
            <circle key={`o${i}`} cx={sx(x)} cy={sy(y)} r={3} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.6} opacity={k === 1 ? 0.5 : 0.85} />
          ))}
          {/* reconstructed (shadow) points */}
          {k === 1 && recon.map(([x, y], i) => (
            <circle key={`s${i}`} cx={sx(x)} cy={sy(y)} r={2.6} fill="var(--accent)" />
          ))}

          {/* legend */}
          <g transform={`translate(18, 24)`} fontFamily="var(--font-sans, sans-serif)" fontSize={10}>
            <circle cx={5} cy={0} r={3} fill="#3c5a8c" /><text x={13} y={3.5} fill="var(--ink-muted)">original</text>
            {k === 1 && <><circle cx={5} cy={15} r={3} fill="var(--accent)" /><text x={13} y={18.5} fill="var(--ink-muted)">reconstructed (1-D)</text>
            <line x1={1} y1={30} x2={9} y2={30} stroke="#c7522a" strokeWidth={1} /><text x={13} y={33.5} fill="var(--ink-muted)">lost (residual)</text></>}
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>variance retained = <span className="text-accent">{(retained * 100).toFixed(1)}%</span></div>
        <div>reconstruction error = <span className="text-ink">{error.toFixed(3)}</span>{k === 1 ? ' (= dropped λ₂)' : ' (exact)'}</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 20.2 — Projection is lossy compression. Keep one component and
        every point collapses onto the <span className="text-accent">PC1</span>{' '}
        line — two numbers become one — and the orange stubs are exactly what is
        thrown away, each point&rsquo;s distance from the line. Because PC1 is the
        widest direction, those residuals are as small as any single line could
        make them, and the variance retained is large. Keep both components and
        the reconstruction is perfect: the error is precisely the variance of the
        dropped direction, no more and no less.
      </figcaption>
    </figure>
  )
}
