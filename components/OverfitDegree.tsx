'use client'

import { useMemo, useState } from 'react'
import { makeCurveData, curveTrue } from '@/lib/foundations'
import { polyFeatures, ridgeRegression, evalPoly } from '@/lib/regularisation'

const W = 480
const H = 340
const PAD = 30
const XR = 1.08
const YR = 1.8

export default function OverfitDegree() {
  const [deg, setDeg] = useState(3)
  const data = useMemo(() => makeCurveData(40, 5, 0.22), [])
  const train = useMemo(() => data.filter((_, i) => i % 4 !== 0), [data])
  const test = useMemo(() => data.filter((_, i) => i % 4 === 0), [data])
  const fit = useMemo(() => ridgeRegression(polyFeatures(train.map((p) => p.x), deg), train.map((p) => p.y), 1e-8), [train, deg])

  const sx = (x: number) => PAD + ((x + XR) / (2 * XR)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + YR) / (2 * YR)) * (H - 2 * PAD)
  const clamp = (y: number) => Math.max(-YR * 1.05, Math.min(YR * 1.05, y))

  const mse = (pts: typeof data) => pts.reduce((s, p) => s + (p.y - evalPoly(fit, p.x)) ** 2, 0) / pts.length
  const trMSE = mse(train), teMSE = mse(test)
  const curve = Array.from({ length: 200 }, (_, k) => { const x = -XR + (k / 199) * 2 * XR; return `${k === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(clamp(evalPoly(fit, x))).toFixed(1)}` }).join(' ')

  const regime = deg <= 1 ? 'underfitting' : deg >= 8 ? 'overfitting — chasing the noise' : 'a good fit'

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">polynomial degree = <span className="font-mono">{deg}</span></span><input type="range" min={1} max={14} step={1} value={deg} onChange={(e) => setDeg(parseInt(e.target.value))} className="w-52" /></label>
        <span className="font-mono text-xs" style={{ color: regime === 'a good fit' ? 'var(--accent)' : '#c7522a' }}>{regime}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="var(--rule)" strokeWidth={0.75} />
          <path d={Array.from({ length: 121 }, (_, k) => { const x = -XR + (k / 120) * 2 * XR; return `${k === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(curveTrue(x)).toFixed(1)}` }).join(' ')} fill="none" stroke="var(--ink-muted)" strokeWidth={1} strokeDasharray="4,4" opacity={0.4} />
          <path d={curve} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
          {train.map((p, i) => <circle key={`tr${i}`} cx={sx(p.x)} cy={sy(p.y)} r={3.2} fill="#3c5a8c" stroke="var(--paper)" strokeWidth={0.6} />)}
          {test.map((p, i) => <circle key={`te${i}`} cx={sx(p.x)} cy={sy(p.y)} r={3.6} fill="none" stroke="#c7522a" strokeWidth={1.6} />)}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>training error = <span className="text-ink">{trMSE.toFixed(3)}</span> <span className="text-ink-faint">(always falls)</span></div>
        <div>test error = <span style={{ color: teMSE > 0.12 ? '#c7522a' : 'var(--accent)' }}>{teMSE.toFixed(3)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 2.2 — Underfitting, good fit, overfitting. Raise the degree and the
        model gains freedom to bend. At low degree it is too stiff to follow the
        trend (high error everywhere). At high degree it threads through every
        training point — driving training error toward zero — but lurches wildly
        between them, and the <span style={{ color: '#c7522a' }}>test error
        explodes</span>: it has memorised the noise. Training error always falls
        with complexity; test error is the one that tells the truth, and it is
        lowest in the middle.
      </figcaption>
    </figure>
  )
}
