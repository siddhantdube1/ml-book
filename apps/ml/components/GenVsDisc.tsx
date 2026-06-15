'use client'

import { useMemo, useState } from 'react'
import { makeHeteroBlobs, fitGaussianNB, gnbPosterior, gnbPredict } from '@/lib/naivebayes'
import { trainLogistic, type LabeledPoint } from '@/lib/logistic'
import type { Point } from '@/lib/tree'

const W = 540
const H = 330
// left scene
const LX0 = 30
const LX1 = 290
const LT = 28
const LB = 300
const X_MIN = -5
const X_MAX = 5
const Y_MIN = -4
const Y_MAX = 4
const GX = 44
const GY = 36
// right learning curve
const RX0 = 350
const RX1 = 520
const RT = 40
const RB = 285

const CLASS = ['#3c5a8c', '#c7522a'] as const
const N_VALUES = [6, 10, 16, 26, 44, 80, 160, 320]
const TEST = 2000

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const alpha = Math.abs(p1 - 0.5) * 2 * 0.45
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

const toLP = (p: Point): LabeledPoint => ({ x: [p.x[0], p.x[1]], y: p.y as 0 | 1 })
function fitLogReg(tr: Point[]) {
  const f = trainLogistic(tr.map(toLP), { lr: 0.2, maxSteps: 250 })
  return f[f.length - 1].model
}
function accNB(m: ReturnType<typeof fitGaussianNB>, d: Point[]) {
  let ok = 0
  for (const p of d) if (gnbPredict(m, p.x) === p.y) ok++
  return ok / d.length
}
function accLR(m: { w: [number, number]; b: number }, d: Point[]) {
  let ok = 0
  for (const p of d) {
    const z = m.b + m.w[0] * p.x[0] + m.w[1] * p.x[1]
    if ((z >= 0 ? 1 : 0) === p.y) ok++
  }
  return ok / d.length
}

export default function GenVsDisc() {
  const [ni, setNi] = useState(2)
  const n = N_VALUES[ni]

  const test = useMemo(() => makeHeteroBlobs(TEST, 9999), [])

  // averaged learning curve (computed once)
  const curve = useMemo(() => {
    const R = 16
    return N_VALUES.map((nv) => {
      let nb = 0, lr = 0
      for (let k = 0; k < R; k++) {
        const tr = makeHeteroBlobs(nv, k * 17 + 1)
        nb += accNB(fitGaussianNB(tr), test)
        lr += accLR(fitLogReg(tr), test)
      }
      return { n: nv, nb: nb / R, lr: lr / R }
    })
  }, [test])

  // the single training set shown in the left scene (seed fixed)
  const train = useMemo(() => makeHeteroBlobs(n, 3), [n])
  const nb = useMemo(() => fitGaussianNB(train), [train])
  const lr = useMemo(() => fitLogReg(train), [train])

  const sx = (x: number) => LX0 + ((x - X_MIN) / (X_MAX - X_MIN)) * (LX1 - LX0)
  const sy = (y: number) => LB - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (LB - LT)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (LX1 - LX0) / GX
    const ch = (LB - LT) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yD = Y_MAX - ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        cells.push({ x: LX0 + i * cw, y: LT + j * ch, w: cw, h: ch, fill: blend(gnbPosterior(nb, [xD, yD])[1]) })
      }
    }
    return cells
  }, [nb])

  const lrLineY = (x: number) => -(lr.w[0] * x + lr.b) / lr.w[1]

  // learning-curve axes
  const cx = (idx: number) => RX0 + (idx / (N_VALUES.length - 1)) * (RX1 - RX0)
  const ACC_MIN = 0.6
  const ACC_MAX = 0.95
  const cy = (a: number) => RB - ((a - ACC_MIN) / (ACC_MAX - ACC_MIN)) * (RB - RT)
  const path = (key: 'nb' | 'lr') =>
    curve.map((d, idx) => `${idx === 0 ? 'M' : 'L'} ${cx(idx).toFixed(1)} ${cy(d[key]).toFixed(1)}`).join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Training points = <span className="font-mono">{n}</span></span>
          <input type="range" min={0} max={N_VALUES.length - 1} step={1} value={ni} onChange={(e) => setNi(parseInt(e.target.value))} className="w-44" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <defs>
            <clipPath id="gd-scene">
              <rect x={LX0} y={LT} width={LX1 - LX0} height={LB - LT} />
            </clipPath>
          </defs>

          {/* left scene */}
          <text x={LX0} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">boundaries on {n} points</text>
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          <g clipPath="url(#gd-scene)">
            <line x1={sx(-6)} y1={sy(lrLineY(-6))} x2={sx(6)} y2={sy(lrLineY(6))} stroke="var(--ink)" strokeWidth={2} strokeDasharray="5,4" />
          </g>
          {train.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={2.6} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.5} />
          ))}

          {/* right learning curve */}
          <text x={RX0} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">test accuracy vs data size</text>
          <line x1={RX0} y1={RB} x2={RX1} y2={RB} stroke="var(--rule)" strokeWidth={1} />
          <line x1={RX0} y1={RT} x2={RX0} y2={RB} stroke="var(--rule)" strokeWidth={1} />
          {[0.6, 0.7, 0.8, 0.9].map((a) => (
            <text key={a} x={RX0 - 5} y={cy(a) + 3} fontSize={9} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{a.toFixed(1)}</text>
          ))}
          {[0, 3, 5, 7].map((idx) => (
            <text key={idx} x={cx(idx)} y={RB + 13} fontSize={9} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{N_VALUES[idx]}</text>
          ))}
          <path d={path('nb')} fill="none" stroke="var(--accent)" strokeWidth={2} />
          <path d={path('lr')} fill="none" stroke="var(--ink)" strokeWidth={1.75} strokeDasharray="5,4" />
          <line x1={cx(ni)} y1={RT} x2={cx(ni)} y2={RB} stroke="var(--ink-faint)" strokeWidth={1} strokeDasharray="2,2" />
          <circle cx={cx(ni)} cy={cy(curve[ni].nb)} r={3.5} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1} />
          <circle cx={cx(ni)} cy={cy(curve[ni].lr)} r={3.5} fill="var(--ink)" stroke="var(--paper)" strokeWidth={1} />
          <g transform={`translate(${RX0 + 6}, ${RT + 2})`}>
            <line x1={0} y1={0} x2={14} y2={0} stroke="var(--accent)" strokeWidth={2} />
            <text x={18} y={3} fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">naive Bayes</text>
            <line x1={0} y1={13} x2={14} y2={13} stroke="var(--ink)" strokeWidth={1.75} strokeDasharray="5,4" />
            <text x={18} y={16} fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">logistic reg.</text>
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>NB test acc = <span className="text-accent">{(accNB(nb, test) * 100).toFixed(1)}%</span> (curved boundary)</div>
        <div>logistic test acc = <span className="text-ink">{(accLR(lr, test) * 100).toFixed(1)}%</span> (straight line)</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 16.4 — Generative versus discriminative. The two classes have
        very different spreads, so the ideal boundary is a closed curve. Gaussian
        naive Bayes <em>models each class</em> and inverts with Bayes&rsquo; rule,
        giving the curved boundary that wraps the tight class; logistic
        regression <em>optimises the split directly</em> and can only draw the
        dashed straight line. Across training sizes (right), NB&rsquo;s generative
        structure lets it use the data more efficiently and pull ahead. Model the
        classes, or model the split — the two faces of the same coin.
      </figcaption>
    </figure>
  )
}
