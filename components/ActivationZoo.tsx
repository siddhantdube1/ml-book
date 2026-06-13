'use client'

import { useMemo, type ReactElement } from 'react'
import { activate, activateDeriv, mlpForward, trainMLP, makeXORData, type Activation, type MLP } from '@/lib/neuralnet'

const W = 680
const H = 430

const ACTS: { kind: Activation; label: string; formula: string }[] = [
  { kind: 'step', label: 'step', formula: '0 or 1' },
  { kind: 'sigmoid', label: 'sigmoid', formula: '1/(1+e⁻ᶻ)' },
  { kind: 'tanh', label: 'tanh', formula: 'tanh z' },
  { kind: 'relu', label: 'ReLU', formula: 'max(0, z)' },
  { kind: 'leaky', label: 'leaky ReLU', formula: 'z or 0.01z' },
]

const RANGE = 4
const ZMIN = -RANGE
const ZMAX = RANGE

function curvePath(f: (z: number) => number, ox: number, oy: number, w: number, h: number, ymin: number, ymax: number): string {
  const pts: string[] = []
  for (let k = 0; k <= 50; k++) {
    const z = ZMIN + (k / 50) * (ZMAX - ZMIN)
    const x = ox + (k / 50) * w
    const y = oy + h - ((f(z) - ymin) / (ymax - ymin)) * h
    pts.push(`${k === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`)
  }
  return pts.join(' ')
}

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1, g = 90 + (82 - 90) * p1, b = 140 + (42 - 140) * p1
  return `rgba(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)},${(Math.abs(p1 - 0.5) * 2 * 0.5).toFixed(3)})`
}

export default function ActivationZoo() {
  // the collapse demo: same XOR net, linear vs ReLU hidden activation
  const xor = useMemo(() => makeXORData(140, 3), [])
  const linNet = useMemo(() => trainMLP(xor, [6], { epochs: 300, lr: 0.3, act: 'linear', seed: 2 }), [xor])
  const reluNet = useMemo(() => trainMLP(xor, [6], { epochs: 400, lr: 0.4, act: 'relu', seed: 2 }), [xor])
  const linLast = linNet[linNet.length - 1]
  const reluLast = reluNet[reluNet.length - 1]

  const miniHeat = (net: MLP, ox: number, oy: number, size: number) => {
    const cells: ReactElement[] = []
    const G = 26
    const cw = size / G
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const xD = -3 + ((i + 0.5) / G) * 6
      const yD = 3 - ((j + 0.5) / G) * 6
      cells.push(<rect key={`${i}-${j}`} x={ox + i * cw} y={oy + j * cw} width={cw} height={cw} fill={blend(mlpForward(net, [xD, yD]).output)} shapeRendering="crispEdges" />)
    }
    return cells
  }
  const sxm = (x: number, ox: number, size: number) => ox + ((x + 3) / 6) * size
  const symm = (y: number, oy: number, size: number) => oy + ((3 - y) / 6) * size

  return (
    <figure className="my-10">
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* activation gallery */}
          <text x={16} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">activation φ(z) (solid) and its derivative φ′(z) (dashed)</text>
          {ACTS.map((a, idx) => {
            const ox = 16 + idx * 132
            const oy = 34
            const pw = 112
            const ph = 86
            const ymin = a.kind === 'tanh' ? -1.1 : -0.1
            const ymax = a.kind === 'relu' || a.kind === 'leaky' ? 4 : 1.1
            return (
              <g key={a.kind}>
                <rect x={ox} y={oy} width={pw} height={ph} fill="var(--paper)" opacity={0.5} stroke="var(--rule)" strokeWidth={0.75} rx={3} />
                {/* zero axes */}
                <line x1={ox} y1={oy + ph - ((0 - ymin) / (ymax - ymin)) * ph} x2={ox + pw} y2={oy + ph - ((0 - ymin) / (ymax - ymin)) * ph} stroke="var(--rule)" strokeWidth={0.5} />
                <line x1={ox + pw / 2} y1={oy} x2={ox + pw / 2} y2={oy + ph} stroke="var(--rule)" strokeWidth={0.5} />
                <path d={curvePath((z) => activateDeriv(a.kind, z), ox, oy, pw, ph, ymin, ymax)} fill="none" stroke="var(--ink-muted)" strokeWidth={1.25} strokeDasharray="3,2" opacity={0.7} />
                <path d={curvePath((z) => activate(a.kind, z), ox, oy, pw, ph, ymin, ymax)} fill="none" stroke="var(--accent)" strokeWidth={2} />
                <text x={ox + 6} y={oy + 14} fontSize={10.5} fill="var(--ink)" fontFamily="var(--font-sans, sans-serif)" fontWeight={600}>{a.label}</text>
                <text x={ox + pw - 5} y={oy + ph - 5} fontSize={8.5} fill="var(--ink-faint)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{a.formula}</text>
              </g>
            )
          })}

          {/* collapse demo */}
          <text x={16} y={166} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">the same 6-unit hidden network on XOR — only the activation differs:</text>
          {([['linear activation → still just a line', linLast, 120], ['ReLU activation → bends to fit', reluLast, 420]] as [string, typeof linLast, number][]).map(([title, frame, ox]) => (
            <g key={title}>
              <text x={ox + 110} y={186} fontSize={10.5} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">{title}</text>
              {miniHeat(frame.net, ox, 196, 220)}
              {xor.map((p, i) => (
                <circle key={i} cx={sxm(p.x[0], ox, 220)} cy={symm(p.x[1], 196, 220)} r={2.4} fill={p.y ? '#c7522a' : '#3c5a8c'} stroke="var(--paper)" strokeWidth={0.4} />
              ))}
              <text x={ox + 110} y={428} fontSize={10} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">accuracy {(frame.accuracy * 100).toFixed(0)}%</text>
            </g>
          ))}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 21.3 — Activations, and why they matter. Each unit passes its
        weighted sum through one of these non-linear bends (top); the dashed
        derivative previews Chapter 22 — note how sigmoid and tanh flatten at the
        edges, where learning crawls, which is why <span className="font-mono">ReLU</span>
        is the modern default. And the non-linearity is the entire point
        (bottom): with a <em>linear</em> activation a six-unit hidden layer still
        draws nothing but a straight line and fails XOR, because a linear
        function of a linear function is linear. Swap in ReLU and the same
        network bends and solves it.
      </figcaption>
    </figure>
  )
}
