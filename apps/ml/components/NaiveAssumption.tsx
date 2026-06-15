'use client'

import { useMemo, useState } from 'react'
import { makeCorrelatedBlobs, fitGaussianNB, gnbPosterior, gnbPredict } from '@/lib/naivebayes'
import type { Point } from '@/lib/tree'

const W = 520
const H = 440
const PAD = 28
const X_MIN = -5.2
const X_MAX = 5.2
const Y_MIN = -4.4
const Y_MAX = 4.4
const GX = 52
const GY = 44

const CLASS = ['#3c5a8c', '#c7522a'] as const

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const alpha = Math.abs(p1 - 0.5) * 2 * 0.4
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

// 2σ ellipse axis vectors from a 2×2 symmetric covariance.
function ellipseAxes(a: number, b: number, c: number): [[number, number], [number, number]] {
  const tr = a + c
  const det = a * c - b * b
  const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0))
  const l1 = tr / 2 + disc
  const l2 = tr / 2 - disc
  let v1: [number, number]
  if (Math.abs(b) > 1e-9) {
    const n = Math.hypot(l1 - c, b)
    v1 = [(l1 - c) / n, b / n]
  } else {
    v1 = a >= c ? [1, 0] : [0, 1]
  }
  const v2: [number, number] = [-v1[1], v1[0]]
  const k = 2 // 2σ
  return [
    [v1[0] * k * Math.sqrt(Math.max(l1, 0)), v1[1] * k * Math.sqrt(Math.max(l1, 0))],
    [v2[0] * k * Math.sqrt(Math.max(l2, 0)), v2[1] * k * Math.sqrt(Math.max(l2, 0))],
  ]
}

export default function NaiveAssumption() {
  const [rho, setRho] = useState(0.7)

  const data = useMemo<Point[]>(() => makeCorrelatedBlobs(200, 11, rho), [rho])
  const model = useMemo(() => fitGaussianNB(data), [data])

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (W - 2 * PAD) / GX
    const ch = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yD = Y_MAX - ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(gnbPosterior(model, [xD, yD])[1]) })
      }
    }
    return cells
  }, [model])

  // per-class true (sample) covariance and NB axis-aligned ellipse, as polygons
  const ellipses = useMemo(() => {
    return [0, 1].map((cls) => {
      const pts = data.filter((p) => p.y === cls)
      const mu = model.mean[cls]
      let sa = 0, sb = 0, sc = 0
      for (const p of pts) {
        const dx = p.x[0] - mu[0]
        const dy = p.x[1] - mu[1]
        sa += dx * dx; sb += dx * dy; sc += dy * dy
      }
      sa /= pts.length; sb /= pts.length; sc /= pts.length
      const trueAx = ellipseAxes(sa, sb, sc)
      const nbAx: [[number, number], [number, number]] = [
        [2 * Math.sqrt(model.variance[cls][0]), 0],
        [0, 2 * Math.sqrt(model.variance[cls][1])],
      ]
      const poly = (ax: [[number, number], [number, number]]) =>
        Array.from({ length: 60 }, (_, k) => {
          const t = (k / 60) * 2 * Math.PI
          const x = mu[0] + Math.cos(t) * ax[0][0] + Math.sin(t) * ax[1][0]
          const y = mu[1] + Math.cos(t) * ax[0][1] + Math.sin(t) * ax[1][1]
          return `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`
        }).join(' ')
      return { cls, truePoly: poly(trueAx), nbPoly: poly(nbAx) }
    })
  }, [data, model])

  const acc = useMemo(() => {
    let ok = 0
    for (const p of data) if (gnbPredict(model, p.x) === p.y) ok++
    return ok / data.length
  }, [data, model])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Correlation ρ = <span className="font-mono">{rho.toFixed(2)}</span></span>
          <input type="range" min={0} max={0.9} step={0.01} value={rho} onChange={(e) => setRho(parseFloat(e.target.value))} className="w-52" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={2.6} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.5} opacity={0.85} />
          ))}
          {ellipses.map((e) => (
            <g key={e.cls}>
              <polygon points={e.nbPoly} fill="none" stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" />
              <polygon points={e.truePoly} fill="none" stroke={CLASS[e.cls]} strokeWidth={2} />
            </g>
          ))}
          {/* legend */}
          <g transform={`translate(${W - 196}, ${PAD + 4})`}>
            <polygon points="0,0 16,0 16,8 0,8" fill="none" stroke="var(--ink)" strokeWidth={2} />
            <text x={22} y={8} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">true (tilted) covariance</text>
            <line x1={0} y1={20} x2={16} y2={20} stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" />
            <text x={22} y={23} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">NB&rsquo;s axis-aligned fit</text>
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>class correlations = <span className="text-ink">±{rho.toFixed(2)}</span> · NB assumes <span className="text-ink">0</span></div>
        <div>NB accuracy = <span className="text-accent">{(acc * 100).toFixed(1)}%</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 16.2 — What &ldquo;naive&rdquo; costs you. Here the two classes
        are told apart by their <em>correlation</em>: each real spread is a
        tilted ellipse, and the classes tilt opposite ways. But a tilt is
        exactly feature correlation, which an axis-aligned Gaussian has none of
        — so naive Bayes fits the <em>upright</em> dashed ellipses, and they come
        out nearly identical for both classes. Turn up ρ and the true ellipses
        cross into an X while NB&rsquo;s stay upright and blind to the
        difference: its accuracy slides as it fits the wrong shape. The
        assumption is a lie; this is the size of the lie.
      </figcaption>
    </figure>
  )
}
