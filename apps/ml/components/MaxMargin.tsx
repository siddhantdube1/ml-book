'use client'

import { useMemo, useState } from 'react'
import type { Point } from '@/lib/tree'
import { makeSvmBlobs, trainKernelSVM, linearWeights, marginWidth } from '@/lib/svm'

const W = 480
const H = 400
const PAD = 26
const X_MIN = -3
const X_MAX = 3
const Y_MIN = -2.5
const Y_MAX = 2.5
const T = 30 // half-length of drawn lines, in data units

const CLASS = ['#3c5a8c', '#c7522a'] as const

type Street = { nx: number; ny: number; c: number; m: number } // unit normal, centre offset, half-width

export default function MaxMargin() {
  const [offsetDeg, setOffsetDeg] = useState(32)
  const [showCandidate, setShowCandidate] = useState(true)

  const data = useMemo<Point[]>(() => makeSvmBlobs(70, 7, 3.0, 0.45), [])

  const svm = useMemo(() => {
    const model = trainKernelSVM(data, { kind: 'linear' }, { C: 10, seed: 3 })
    const { w, b } = linearWeights(model)
    const norm = Math.sqrt(w[0] * w[0] + w[1] * w[1])
    const street: Street = {
      nx: w[0] / norm,
      ny: w[1] / norm,
      c: -b / norm,
      m: marginWidth(w),
    }
    const sv = data.filter((_, i) => model.alpha[i] > 1e-6)
    const theta = Math.atan2(w[1], w[0])
    return { street, sv, theta }
  }, [data])

  // Candidate boundary: a line of normal angle (svm angle + offset), placed
  // at the widest-street position for that orientation.
  const candidate = useMemo<Street & { separable: boolean }>(() => {
    const theta = svm.theta + (offsetDeg * Math.PI) / 180
    const nx = Math.cos(theta)
    const ny = Math.sin(theta)
    const proj = data.map((p) => ({ v: nx * p.x[0] + ny * p.x[1], y: p.y }))
    const m0 = proj.filter((p) => p.y === 0)
    const m1 = proj.filter((p) => p.y === 1)
    const mean0 = m0.reduce((s, p) => s + p.v, 0) / m0.length
    const mean1 = m1.reduce((s, p) => s + p.v, 0) / m1.length
    const pos = mean1 >= mean0 ? m1 : m0
    const neg = mean1 >= mean0 ? m0 : m1
    const minPos = Math.min(...pos.map((p) => p.v))
    const maxNeg = Math.max(...neg.map((p) => p.v))
    const c = (minPos + maxNeg) / 2
    const m = (minPos - maxNeg) / 2
    return { nx, ny, c, m: Math.abs(m), separable: m > 0 }
  }, [data, svm.theta, offsetDeg])

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD)

  // A line at offset `off` along the normal, as two box-spanning endpoints.
  function linePts(s: Street, off: number): [number, number, number, number] {
    const px = (s.c + off) * s.nx
    const py = (s.c + off) * s.ny
    const dx = -s.ny
    const dy = s.nx
    return [px - T * dx, py - T * dy, px + T * dx, py + T * dy]
  }
  function streetPoly(s: Street): string {
    const a = linePts(s, s.m)
    const b = linePts(s, -s.m)
    return [
      `${sx(a[0])},${sy(a[1])}`,
      `${sx(a[2])},${sy(a[3])}`,
      `${sx(b[2])},${sy(b[3])}`,
      `${sx(b[0])},${sy(b[1])}`,
    ].join(' ')
  }
  function lineEl(s: Street, off: number, stroke: string, dash?: string, sw = 1.5) {
    const [x1, y1, x2, y2] = linePts(s, off)
    return (
      <line
        x1={sx(x1)}
        y1={sy(y1)}
        x2={sx(x2)}
        y2={sy(y2)}
        stroke={stroke}
        strokeWidth={sw}
        strokeDasharray={dash}
      />
    )
  }

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Tilt candidate = <span className="font-mono">{offsetDeg}°</span></span>
          <input type="range" min={-60} max={60} step={1} value={offsetDeg} onChange={(e) => setOffsetDeg(parseInt(e.target.value))} className="w-44" />
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={showCandidate} onChange={(e) => setShowCandidate(e.target.checked)} />
          <span className="text-ink-muted">show candidate line</span>
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <defs>
            <clipPath id="mm-plot">
              <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} />
            </clipPath>
          </defs>
          <g clipPath="url(#mm-plot)">
            {/* SVM maximum-margin street */}
            <polygon points={streetPoly(svm.street)} fill="var(--accent)" opacity={0.1} />
            {lineEl(svm.street, svm.street.m, 'var(--accent)', '5,4', 1.25)}
            {lineEl(svm.street, -svm.street.m, 'var(--accent)', '5,4', 1.25)}
            {lineEl(svm.street, 0, 'var(--accent)', undefined, 2.25)}

            {/* candidate line + its best street for this orientation */}
            {showCandidate && candidate.separable && (
              <polygon points={streetPoly(candidate)} fill="var(--ink-muted)" opacity={0.08} />
            )}
            {showCandidate && (
              <>
                {lineEl(candidate, 0, 'var(--ink-muted)', '2,3', 1.75)}
                {candidate.separable && lineEl(candidate, candidate.m, 'var(--ink-muted)', '2,3', 0.9)}
                {candidate.separable && lineEl(candidate, -candidate.m, 'var(--ink-muted)', '2,3', 0.9)}
              </>
            )}
          </g>

          {/* support vectors: ring behind the point */}
          {svm.sv.map((p, i) => (
            <circle key={`sv${i}`} cx={sx(p.x[0])} cy={sy(p.x[1])} r={7} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
          ))}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3.5} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.7} />
          ))}

          <text x={PAD} y={PAD - 9} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            teal = maximum-margin SVM · ringed = support vectors
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>SVM margin = <span className="text-accent">{svm.street.m.toFixed(3)}</span> · support vectors = <span className="text-ink">{svm.sv.length}</span></div>
        <div>
          candidate margin ={' '}
          <span className="text-ink">{candidate.separable ? candidate.m.toFixed(3) : 'overlaps'}</span>
          {candidate.separable && candidate.m < svm.street.m - 1e-3 && (
            <span className="text-ink-faint"> ({((1 - candidate.m / svm.street.m) * 100).toFixed(0)}% narrower)</span>
          )}
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 15.1 — The widest street. Of all the lines that separate the two
        classes, the support vector machine (teal) picks the one with the
        largest margin — the widest band you can lay between the classes before
        a point falls in. Tilt the grey candidate to any other orientation and
        its best-possible street is always narrower. Only the ringed points —
        the <em>support vectors</em>, pressed against the margin — touch the
        SVM&rsquo;s street; every other point could move freely without shifting
        the boundary at all.
      </figcaption>
    </figure>
  )
}
