'use client'

import { useMemo, useState } from 'react'
import { makeCurveData } from '@/lib/foundations'
import { polyFeatures, ridgeRegression, evalPoly } from '@/lib/regularisation'

const W = 520
const H = 340
const PAD_L = 48
const PAD_R = 16
const PAD_T = 24
const PAD_B = 42
const MAX_DEG = 11
const Y_MAX = 0.18

export default function ComplexityCurve() {
  const [deg, setDeg] = useState(3)

  // average train/test error over several noise realisations for clean curves
  const curves = useMemo(() => {
    const train: number[] = []
    const test: number[] = []
    const R = 12
    for (let d = 1; d <= MAX_DEG; d++) {
      let tr = 0, te = 0
      for (let s = 0; s < R; s++) {
        const data = makeCurveData(40, s + 1, 0.22)
        const trn = data.filter((_, i) => i % 4 !== 0)
        const tst = data.filter((_, i) => i % 4 === 0)
        const w = ridgeRegression(polyFeatures(trn.map((p) => p.x), d), trn.map((p) => p.y), 1e-8)
        tr += trn.reduce((s2, p) => s2 + (p.y - evalPoly(w, p.x)) ** 2, 0) / trn.length
        te += tst.reduce((s2, p) => s2 + (p.y - evalPoly(w, p.x)) ** 2, 0) / tst.length
      }
      train.push(tr / R)
      test.push(te / R)
    }
    return { train, test, best: test.indexOf(Math.min(...test)) + 1 }
  }, [])

  const dx = (d: number) => PAD_L + ((d - 1) / (MAX_DEG - 1)) * (W - PAD_L - PAD_R)
  const dy = (e: number) => H - PAD_B - (Math.min(e, Y_MAX) / Y_MAX) * (H - PAD_T - PAD_B)
  const path = (arr: number[]) => arr.map((e, i) => `${i === 0 ? 'M' : 'L'} ${dx(i + 1).toFixed(1)} ${dy(e).toFixed(1)}`).join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">complexity (degree) = <span className="font-mono">{deg}</span></span><input type="range" min={1} max={MAX_DEG} step={1} value={deg} onChange={(e) => setDeg(parseInt(e.target.value))} className="w-44" /></label>
        <span className="font-mono text-xs text-accent">best degree ≈ {curves.best}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />
          {/* sweet-spot marker */}
          <line x1={dx(curves.best)} y1={PAD_T} x2={dx(curves.best)} y2={H - PAD_B} stroke="var(--accent)" strokeWidth={1} strokeDasharray="3,3" opacity={0.5} />
          {/* current-degree marker */}
          <line x1={dx(deg)} y1={PAD_T} x2={dx(deg)} y2={H - PAD_B} stroke="var(--ink-faint)" strokeWidth={1} strokeDasharray="2,2" />

          <path d={path(curves.train)} fill="none" stroke="var(--ink-muted)" strokeWidth={1.75} strokeDasharray="4,3" />
          <path d={path(curves.test)} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
          <circle cx={dx(deg)} cy={dy(curves.test[deg - 1])} r={4} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1} />
          <circle cx={dx(deg)} cy={dy(curves.train[deg - 1])} r={3.5} fill="var(--ink-muted)" stroke="var(--paper)" strokeWidth={1} />

          {/* labels */}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 8} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">model complexity →</text>
          <text x={PAD_L} y={PAD_T - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">error</text>
          <text x={W - PAD_R - 6} y={dy(0.02) + 3} fontSize={10} fill="var(--ink-faint)" textAnchor="end" fontFamily="var(--font-sans, sans-serif)">underfit ← → overfit</text>
          <g transform={`translate(${PAD_L + 14}, ${PAD_T + 6})`} fontFamily="var(--font-sans, sans-serif)" fontSize={10}>
            <line x1={0} y1={0} x2={16} y2={0} stroke="var(--accent)" strokeWidth={2.25} /><text x={21} y={3.5} fill="var(--ink-muted)">test error</text>
            <line x1={0} y1={14} x2={16} y2={14} stroke="var(--ink-muted)" strokeWidth={1.75} strokeDasharray="4,3" /><text x={21} y={17.5} fill="var(--ink-muted)">training error</text>
          </g>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 2.3 — The U that governs everything. Plot error against model
        complexity and the two curves split. Training error (grey) only ever falls:
        more flexibility always fits the training data better. Test error (teal)
        falls, bottoms out, and climbs — high on the left where the model is too
        simple (underfit), high on the right where it has memorised noise
        (overfit), lowest in the sweet spot between. Choosing a model is choosing
        where to sit on this curve, and the test error is the only honest guide.
      </figcaption>
    </figure>
  )
}
