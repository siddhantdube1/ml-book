'use client'

import { useMemo, useState } from 'react'
import { makeBlobs2, type Point } from '@/lib/tree'
import { fitGaussianNB, gnbPosterior } from '@/lib/naivebayes'

const W = 520
const H = 470
// main plot
const ML = 44
const MR = 458
const MT = 70
const MB = 410
// marginal strips
const TOP_H = 52 // x1-feature densities along the top
const RIGHT_W = 52 // x2-feature densities along the right
const X_MIN = -3.4
const X_MAX = 3.4
const Y_MIN = -2.8
const Y_MAX = 2.8
const GX = 56
const GY = 50

const CLASS = ['#3c5a8c', '#c7522a'] as const

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const alpha = Math.abs(p1 - 0.5) * 2 * 0.5
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

const gauss1 = (x: number, mu: number, v: number) =>
  Math.exp(-((x - mu) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)

export default function ClassConditional() {
  const [prior1, setPrior1] = useState(0.5)

  const data = useMemo<Point[]>(() => makeBlobs2(160, 7, 2), [])
  const base = useMemo(() => fitGaussianNB(data), [data])
  const model = useMemo(
    () => ({ ...base, prior: [1 - prior1, prior1] }),
    [base, prior1],
  )

  const sx = (x: number) => ML + ((x - X_MIN) / (X_MAX - X_MIN)) * (MR - ML)
  const sy = (y: number) => MB - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (MB - MT)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (MR - ML) / GX
    const ch = (MB - MT) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yD = Y_MAX - ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        const p1 = gnbPosterior(model, [xD, yD])[1]
        cells.push({ x: ML + i * cw, y: MT + j * ch, w: cw, h: ch, fill: blend(p1) })
      }
    }
    return cells
  }, [model])

  // marginal densities (top: feature x1; right: feature x2)
  const topPaths = [0, 1].map((c) => {
    const mu = base.mean[c][0]
    const v = base.variance[c][0]
    const peak = gauss1(mu, mu, v)
    const pts: string[] = []
    for (let k = 0; k <= 80; k++) {
      const xD = X_MIN + (k / 80) * (X_MAX - X_MIN)
      const d = gauss1(xD, mu, v) / peak
      pts.push(`${k === 0 ? 'M' : 'L'} ${sx(xD).toFixed(1)} ${(TOP_H + 12 - d * (TOP_H - 6)).toFixed(1)}`)
    }
    return pts.join(' ')
  })
  const rightPaths = [0, 1].map((c) => {
    const mu = base.mean[c][1]
    const v = base.variance[c][1]
    const peak = gauss1(mu, mu, v)
    const pts: string[] = []
    for (let k = 0; k <= 80; k++) {
      const yD = Y_MIN + (k / 80) * (Y_MAX - Y_MIN)
      const d = gauss1(yD, mu, v) / peak
      pts.push(`${k === 0 ? 'M' : 'L'} ${(MR + 8 + d * (RIGHT_W - 8)).toFixed(1)} ${sy(yD).toFixed(1)}`)
    }
    return pts.join(' ')
  })

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Prior P(orange) = <span className="font-mono">{prior1.toFixed(2)}</span></span>
          <input type="range" min={0.1} max={0.9} step={0.01} value={prior1} onChange={(e) => setPrior1(parseFloat(e.target.value))} className="w-48" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}

          {/* top marginal densities: P(x1 | class) */}
          {topPaths.map((d, c) => (
            <path key={`t${c}`} d={d} fill="none" stroke={CLASS[c]} strokeWidth={1.5} opacity={0.85} />
          ))}
          <text x={ML} y={10} fontSize={9.5} fill="var(--ink-faint)" fontFamily="var(--font-mono, monospace)">P(x₁ | class)</text>
          {/* right marginal densities: P(x2 | class) */}
          {rightPaths.map((d, c) => (
            <path key={`r${c}`} d={d} fill="none" stroke={CLASS[c]} strokeWidth={1.5} opacity={0.85} />
          ))}

          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.7} />
          ))}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 16.1 — Gaussian naive Bayes. Each class is modelled as a product
        of one Gaussian <em>per feature</em> — the bell curves along the top
        (for x₁) and right (for x₂). Multiply those per-feature likelihoods,
        weight by the class prior, and normalise, and you get the posterior
        shaded here; the boundary falls where it crosses ½. Slide the prior and
        the whole boundary shifts: a rarer class must clear a higher bar of
        evidence before it is predicted — Bayes&rsquo; rule, made to move.
      </figcaption>
    </figure>
  )
}
