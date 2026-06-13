'use client'

import { useMemo, useState } from 'react'
import { make1DLinear, leastSquaresLine, rSquared } from '@/lib/foundations'

const W = 680
const H = 320
const X0 = 0, X1 = 10, Y0 = -1, Y1 = 11

export default function ResidualsR2() {
  const [noise, setNoise] = useState(0.8)
  const data = useMemo(() => make1DLinear(40, 3, 0.8, 1.8, noise, 0.5, 9.5), [noise])
  const best = useMemo(() => leastSquaresLine(data.map((p) => p.x), data.map((p) => p.y)), [data])
  const r2 = rSquared(data, best.slope, best.intercept)
  const line = (x: number) => best.slope * x + best.intercept

  // left panel: data + line
  const L0 = 8, LW = 312
  const lsx = (x: number) => L0 + 14 + ((x - X0) / (X1 - X0)) * (LW - 28)
  const lsy = (y: number) => H - 28 - ((y - Y0) / (Y1 - Y0)) * (H - 50)
  // right panel: residuals vs x
  const R0 = 356, RW = 316
  const resid = data.map((p) => p.y - line(p.x))
  const rmax = Math.max(1, ...resid.map((r) => Math.abs(r))) * 1.1
  const rsx = (x: number) => R0 + 14 + ((x - X0) / (X1 - X0)) * (RW - 28)
  const rsy = (r: number) => H / 2 - (r / rmax) * (H - 70) / 2

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">noise = <span className="font-mono">{noise.toFixed(1)}</span></span><input type="range" min={0.1} max={4} step={0.1} value={noise} onChange={(e) => setNoise(parseFloat(e.target.value))} className="w-48" /></label>
        <span className="font-mono text-xs">R² = <span style={{ color: r2 > 0.7 ? 'var(--accent)' : '#c7522a' }}>{r2.toFixed(3)}</span></span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <text x={L0 + LW / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">data and best-fit line</text>
          <text x={R0 + RW / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">residuals (miss vs x)</text>
          <line x1={340} y1={28} x2={340} y2={H - 14} stroke="var(--rule)" strokeWidth={1} />

          {/* left */}
          <line x1={lsx(X0)} y1={lsy(line(X0))} x2={lsx(X1)} y2={lsy(line(X1))} stroke="var(--accent)" strokeWidth={2.25} />
          {data.map((p, i) => <circle key={`l${i}`} cx={lsx(p.x)} cy={lsy(p.y)} r={2.8} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.5} />)}

          {/* right: residual plot */}
          <line x1={rsx(X0)} y1={rsy(0)} x2={rsx(X1)} y2={rsy(0)} stroke="var(--rule)" strokeWidth={1.25} />
          {data.map((p, i) => (
            <g key={`r${i}`}>
              <line x1={rsx(p.x)} y1={rsy(0)} x2={rsx(p.x)} y2={rsy(resid[i])} stroke="#c7522a" strokeWidth={0.7} opacity={0.4} />
              <circle cx={rsx(p.x)} cy={rsy(resid[i])} r={2.6} fill="#c7522a" opacity={0.8} />
            </g>
          ))}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 5.3 — How good is the fit? <strong>R²</strong> is the fraction of the
        data&rsquo;s variation the line explains — 1 for a perfect fit, 0 for a line no
        better than guessing the mean. Turn up the noise and the points scatter further
        from any line, the residuals (right) grow, and R² falls toward zero: the same
        straight-line *model* now explains far less, because the data simply has more
        unexplainable scatter. A good residual plot, by the way, looks like
        structureless noise around zero — any pattern there means the straight line
        missed something.
      </figcaption>
    </figure>
  )
}
