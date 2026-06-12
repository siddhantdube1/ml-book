'use client'

import { useMemo, useState } from 'react'
import { makeMoons2, type Point } from '@/lib/tree'
import { trainKernelSVM, kernelDecision } from '@/lib/svm'

const PW = 440
const PH = 360
const PAD = 24
const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.6
const Y_MAX = 2.6
const GX = 60
const GY = 50

const G_MIN = 0.08
const G_MAX = 20
const C_MIN = 0.2
const C_MAX = 20

const CLASS = ['#3c5a8c', '#c7522a'] as const

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const certainty = Math.abs(p1 - 0.5) * 2
  const alpha = certainty * 0.5
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

export default function KernelBoundary() {
  const [tg, setTg] = useState(0.5) // -> log gamma
  const [tc, setTc] = useState(0.4) // -> log C

  const gamma = useMemo(() => Math.pow(10, Math.log10(G_MIN) + tg * (Math.log10(G_MAX) - Math.log10(G_MIN))), [tg])
  const C = useMemo(() => Math.pow(10, Math.log10(C_MIN) + tc * (Math.log10(C_MAX) - Math.log10(C_MIN))), [tc])

  const train = useMemo<Point[]>(() => makeMoons2(140, 1, 0.2, 0.12), [])
  const val = useMemo<Point[]>(() => makeMoons2(140, 77, 0.2, 0.12), [])

  const model = useMemo(
    () => trainKernelSVM(train, { kind: 'rbf', gamma }, { C, seed: 3, maxPasses: 20 }),
    [train, gamma, C],
  )

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (PW - 2 * PAD)
  const sy = (y: number) => PH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (PH - 2 * PAD)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (PW - 2 * PAD) / GX
    const ch = (PH - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yD = Y_MAX - ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        const p1 = (Math.tanh(kernelDecision(model, [xD, yD])) + 1) / 2
        cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(p1) })
      }
    }
    return cells
  }, [model])

  const stats = useMemo(() => {
    const a = (d: Point[]) => {
      let ok = 0
      for (const p of d) if ((kernelDecision(model, p.x) >= 0 ? 1 : 0) === p.y) ok++
      return ok / d.length
    }
    return { tr: a(train), va: a(val), nSV: model.alpha.filter((x) => x > 1e-6).length }
  }, [model, train, val])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Kernel width γ = <span className="font-mono">{gamma.toFixed(2)}</span></span>
          <input type="range" min={0} max={1} step={0.01} value={tg} onChange={(e) => setTg(parseFloat(e.target.value))} className="w-40" />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Penalty C = <span className="font-mono">{C < 1 ? C.toFixed(2) : C.toFixed(1)}</span></span>
          <input type="range" min={0} max={1} step={0.01} value={tc} onChange={(e) => setTc(parseFloat(e.target.value))} className="w-32" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          {train.map((p, i) =>
            model.alpha[i] > 1e-6 ? (
              <circle key={`sv${i}`} cx={sx(p.x[0])} cy={sy(p.x[1])} r={6.5} fill="none" stroke="var(--ink)" strokeWidth={1.2} />
            ) : null,
          )}
          {train.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.7} />
          ))}
          <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            RBF-kernel SVM · ringed = support vectors
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>train acc = <span className="text-ink">{(stats.tr * 100).toFixed(1)}%</span></div>
        <div>validation acc = <span className="text-accent">{(stats.va * 100).toFixed(1)}%</span></div>
        <div>support vectors = <span className="text-ink">{stats.nSV}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 15.5 — The RBF-kernel SVM in practice, on the moons of Chapters
        13&ndash;14. The Gaussian kernel&rsquo;s width{' '}
        <span className="font-mono">γ</span> is a complexity knob: small{' '}
        <span className="font-mono">γ</span> gives a smooth, nearly straight
        boundary that underfits; large{' '}
        <span className="font-mono">γ</span> wraps tight islands around
        individual points — training accuracy climbs while validation accuracy
        falls and the support-vector count balloons, the unmistakable signature
        of overfitting. Between them lies the sweet spot, found exactly as in
        Chapter 11. Set this beside the forest and boosting boundaries on the
        same data: three very different machines carving the same two arcs.
      </figcaption>
    </figure>
  )
}
