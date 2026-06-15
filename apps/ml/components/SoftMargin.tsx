'use client'

import { useMemo, useState } from 'react'
import type { Point } from '@/lib/tree'
import {
  makeSvmBlobs,
  trainKernelSVM,
  linearWeights,
  marginWidth,
} from '@/lib/svm'

const W = 480
const H = 400
const PAD = 26
const X_MIN = -3
const X_MAX = 3
const Y_MIN = -2.5
const Y_MAX = 2.5
const T = 30
const C_MIN = 0.04
const C_MAX = 40

const CLASS = ['#3c5a8c', '#c7522a'] as const

export default function SoftMargin() {
  const [t, setT] = useState(0.38) // -> log C, default ~0.4

  const C = useMemo(() => {
    const lo = Math.log10(C_MIN)
    const hi = Math.log10(C_MAX)
    return Math.pow(10, lo + t * (hi - lo))
  }, [t])

  const data = useMemo<Point[]>(() => makeSvmBlobs(110, 11, 1.7, 0.62), [])

  const fit = useMemo(() => {
    const model = trainKernelSVM(data, { kind: 'linear' }, { C, seed: 3 })
    const { w, b } = linearWeights(model)
    const norm = Math.sqrt(w[0] * w[0] + w[1] * w[1])
    const street = { nx: w[0] / norm, ny: w[1] / norm, c: -b / norm, m: marginWidth(w) }
    let ok = 0
    let violators = 0
    const isSV: boolean[] = []
    data.forEach((p, i) => {
      const dec = w[0] * p.x[0] + w[1] * p.x[1] + b
      const yy = p.y === 1 ? 1 : -1
      if ((dec >= 0 ? 1 : 0) === p.y) ok++
      if (yy * dec < 1 - 1e-6) violators++
      isSV.push(model.alpha[i] > 1e-6)
    })
    return { street, isSV, acc: ok / data.length, violators, nSV: model.alpha.filter((a) => a > 1e-6).length }
  }, [data, C])

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD)

  const s = fit.street
  const linePts = (off: number): [number, number, number, number] => {
    const px = (s.c + off) * s.nx
    const py = (s.c + off) * s.ny
    const dx = -s.ny
    const dy = s.nx
    return [px - T * dx, py - T * dy, px + T * dx, py + T * dy]
  }
  const streetPoly = (): string => {
    const a = linePts(s.m)
    const b = linePts(-s.m)
    return [`${sx(a[0])},${sy(a[1])}`, `${sx(a[2])},${sy(a[3])}`, `${sx(b[2])},${sy(b[3])}`, `${sx(b[0])},${sy(b[1])}`].join(' ')
  }
  const lineEl = (off: number, dash: string | undefined, sw: number) => {
    const [x1, y1, x2, y2] = linePts(off)
    return <line x1={sx(x1)} y1={sy(y1)} x2={sx(x2)} y2={sy(y2)} stroke="var(--accent)" strokeWidth={sw} strokeDasharray={dash} />
  }

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Penalty C = <span className="font-mono">{C < 1 ? C.toFixed(2) : C.toFixed(1)}</span></span>
          <input type="range" min={0} max={1} step={0.01} value={t} onChange={(e) => setT(parseFloat(e.target.value))} className="w-52" />
        </label>
        <span className="font-mono text-xs text-ink-faint">small C = wide, soft · large C = narrow, hard</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <defs>
            <clipPath id="sm-plot">
              <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} />
            </clipPath>
          </defs>
          <g clipPath="url(#sm-plot)">
            <polygon points={streetPoly()} fill="var(--accent)" opacity={0.1} />
            {lineEl(s.m, '5,4', 1.25)}
            {lineEl(-s.m, '5,4', 1.25)}
            {lineEl(0, undefined, 2.25)}
          </g>

          {data.map((p, i) =>
            fit.isSV[i] ? (
              <circle key={`sv${i}`} cx={sx(p.x[0])} cy={sy(p.x[1])} r={7} fill="none" stroke="var(--ink)" strokeWidth={1.4} />
            ) : null,
          )}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3.5} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.7} />
          ))}

          <text x={PAD} y={PAD - 9} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            ringed = support vectors (points on or inside the margin)
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>margin = <span className="text-accent">{s.m.toFixed(3)}</span></div>
        <div>support vectors = <span className="text-ink">{fit.nSV}</span></div>
        <div>training accuracy = <span className="text-ink">{(fit.acc * 100).toFixed(1)}%</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 15.2 — The soft margin. When the classes overlap, no line
        separates them, so the SVM allows points to sit inside the margin — or
        on the wrong side — and pays a penalty <span className="font-mono">C</span> for
        each. A large <span className="font-mono">C</span> punishes every
        violation, forcing a narrow, hard margin that contorts to catch points;
        a small <span className="font-mono">C</span> buys a wide, smooth margin
        at the cost of more violations and more support vectors. It is the
        bias&ndash;variance dial of Chapters 9&ndash;11 in the SVM&rsquo;s clothing.
      </figcaption>
    </figure>
  )
}
