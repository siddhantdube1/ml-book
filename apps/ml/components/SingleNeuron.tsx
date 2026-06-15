'use client'

import { useMemo, useState } from 'react'
import { activate, type Activation } from '@/lib/neuralnet'

const W = 400
const H = 400
const PAD = 24
const RANGE = 3.2
const GX = 48
const GY = 48

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const alpha = Math.abs(p1 - 0.5) * 2 * 0.5
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

export default function SingleNeuron() {
  const [w1, setW1] = useState(1.2)
  const [w2, setW2] = useState(0.8)
  const [b, setB] = useState(-0.3)
  const [act, setAct] = useState<Activation>('sigmoid')

  const sx = (x: number) => PAD + ((x + RANGE) / (2 * RANGE)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + RANGE) / (2 * RANGE)) * (H - 2 * PAD)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (W - 2 * PAD) / GX
    const ch = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = -RANGE + ((i + 0.5) / GX) * 2 * RANGE
        const yD = RANGE - ((j + 0.5) / GY) * 2 * RANGE
        const out = activate(act, w1 * xD + w2 * yD + b)
        cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(out) })
      }
    }
    return cells
  }, [w1, w2, b, act])

  // boundary line w1*x + w2*y + b = 0
  const lineY = (x: number) => -(w1 * x + b) / w2

  // activation curve inset
  const curve = useMemo(() => {
    const pts: string[] = []
    for (let k = 0; k <= 60; k++) {
      const z = -6 + (k / 60) * 12
      const v = activate(act, z)
      pts.push(`${k === 0 ? 'M' : 'L'} ${(8 + (z + 6) / 12 * 96).toFixed(1)} ${(96 - v * 64).toFixed(1)}`)
    }
    return pts.join(' ')
  }, [act])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">w₁ = <span className="font-mono">{w1.toFixed(1)}</span></span><input type="range" min={-3} max={3} step={0.1} value={w1} onChange={(e) => setW1(parseFloat(e.target.value))} className="w-28" /></label>
        <label className="flex items-center gap-2"><span className="text-ink-muted">w₂ = <span className="font-mono">{w2.toFixed(1)}</span></span><input type="range" min={-3} max={3} step={0.1} value={w2} onChange={(e) => setW2(parseFloat(e.target.value))} className="w-28" /></label>
        <label className="flex items-center gap-2"><span className="text-ink-muted">b = <span className="font-mono">{b.toFixed(1)}</span></span><input type="range" min={-3} max={3} step={0.1} value={b} onChange={(e) => setB(parseFloat(e.target.value))} className="w-28" /></label>
        <div className="flex rounded border border-rule overflow-hidden">
          {(['step', 'sigmoid'] as const).map((a) => (
            <button key={a} onClick={() => setAct(a)} className="px-2.5 py-1 transition-colors" style={{ background: act === a ? 'var(--accent)' : 'transparent', color: act === a ? 'var(--paper)' : 'var(--ink-muted)' }}>{a}</button>
          ))}
        </div>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          {Math.abs(w2) > 1e-6 && (
            <line x1={sx(-RANGE)} y1={sy(lineY(-RANGE))} x2={sx(RANGE)} y2={sy(lineY(RANGE))} stroke="var(--accent)" strokeWidth={2} />
          )}
          {/* activation inset */}
          <g transform={`translate(${W - 122}, ${PAD + 6})`}>
            <rect x={0} y={20} width={112} height={84} fill="var(--paper)" opacity={0.82} rx={3} stroke="var(--rule)" strokeWidth={0.75} />
            <line x1={8} y1={96} x2={104} y2={96} stroke="var(--rule)" strokeWidth={0.75} />
            <line x1={56} y1={28} x2={56} y2={100} stroke="var(--rule)" strokeWidth={0.5} />
            <path d={curve} fill="none" stroke="var(--accent)" strokeWidth={1.75} />
            <text x={8} y={18} fontSize={9} fill="var(--ink-muted)" fontFamily="var(--font-mono, monospace)">φ(w·x + b)</text>
          </g>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 21.1 — A single neuron. It multiplies each input by a weight, adds
        a bias, and passes the sum through an activation φ — and that is a linear
        classifier: the weights set the tilt of the boundary, the bias shifts it.
        The <span className="font-mono">step</span> activation gives a hard
        decision (the perceptron); the <span className="font-mono">sigmoid</span>
        softens it into a probability — at which point this neuron is precisely
        the logistic regression of Chapter 7. One neuron, one straight line.
      </figcaption>
    </figure>
  )
}
