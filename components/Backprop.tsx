'use client'

import { useEffect, useMemo, useState } from 'react'
import { backpropTrace, type MLP } from '@/lib/neuralnet'

const W = 540
const H = 300
const FWD = 'var(--accent)'
const BWD = '#c7522a'

// a fixed 2-2-1 network and one (mis-classified) example, so gradients are real
const NET: MLP = {
  W: [
    [[1.0, 0.8], [-0.6, 1.2]],
    [[1.1, -0.9]],
  ],
  b: [[0.1, -0.3], [0.2]],
  act: 'tanh',
}
const X = [1.2, -0.8]
const Y = 1 as const

const IN = [{ x: 70, y: 90 }, { x: 70, y: 210 }]
const HID = [{ x: 250, y: 80 }, { x: 250, y: 220 }]
const OUT = { x: 430, y: 150 }

const STAGES = [
  'forward — feed the inputs in',
  'forward — each hidden unit: weighted sum, then activation',
  'forward — the output, and the loss',
  'backward — the error at the output: ŷ − y',
  'backward — push the error back through the weights × φ′',
  'gradients — backward error × forward signal, at every weight',
] as const

export default function Backprop() {
  const [stage, setStage] = useState(0)
  const [playing, setPlaying] = useState(false)
  const t = useMemo(() => backpropTrace(NET, X, Y), [])

  useEffect(() => {
    if (!playing) return
    if (stage >= STAGES.length - 1) { setPlaying(false); return }
    const id = setTimeout(() => setStage((s) => s + 1), 1300)
    return () => clearTimeout(id)
  }, [playing, stage])

  const fwd = stage >= 0
  const showHid = stage >= 1
  const showOut = stage >= 2
  const showOutDelta = stage >= 3
  const showHidDelta = stage >= 4
  const showGrad = stage >= 5

  const n2 = (v: number) => v.toFixed(2)

  // worked equation for the current stage (real numbers from the trace)
  const eq = (() => {
    switch (stage) {
      case 0: return `inputs  x = (${n2(X[0])}, ${n2(X[1])})`
      case 1: return `h₁ = φ(w·x + b) = tanh(${n2(t.z[0][0])}) = ${n2(t.a[1][0])}`
      case 2: return `ŷ = σ(${n2(t.z[1][0])}) = ${n2(t.output)}   ·   loss = ${n2(t.loss)}   (target y = ${Y})`
      case 3: return `δ_out = ŷ − y = ${n2(t.output)} − ${Y} = ${n2(t.delta[1][0])}`
      case 4: return `δ_h₁ = (w · δ_out) · φ′(z) = ${n2(t.delta[0][0])}   (the error, propagated back)`
      case 5: return `∂L/∂w = δ_out · a_h₁ = ${n2(t.delta[1][0])} · ${n2(t.a[1][0])} = ${n2(t.gradW[1][0][0])}`
      default: return ''
    }
  })()

  const edge = (a: { x: number; y: number }, b: { x: number; y: number }, key: string, on: boolean, col: string, label?: string) => (
    <g key={key}>
      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={on ? col : 'var(--rule)'} strokeWidth={on ? 1.6 : 1} opacity={on ? 0.85 : 0.5} />
      {label && on && (
        <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 3} fontSize={9} fill={col} textAnchor="middle" fontFamily="var(--font-mono, monospace)">{label}</text>
      )}
    </g>
  )

  const node = (p: { x: number; y: number }, val: string | null, col: string, sub?: string | null) => (
    <g>
      <circle cx={p.x} cy={p.y} r={20} fill="var(--paper)" stroke={val ? col : 'var(--rule)'} strokeWidth={val ? 2 : 1} />
      {val && <text x={p.x} y={p.y + 4} fontSize={12} fill="var(--ink)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{val}</text>}
      {sub && <text x={p.x} y={p.y + 33} fontSize={9} fill={col} textAnchor="middle" fontFamily="var(--font-mono, monospace)">{sub}</text>}
    </g>
  )

  return (
    <figure className="my-10">
      <div className="font-sans text-sm text-ink-muted mb-3">{STAGES[stage]}</div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* edges input -> hidden */}
          {IN.map((ip, i) => HID.map((hp, h) => edge(ip, hp, `ih${i}${h}`,
            showGrad ? true : showHid, showGrad ? BWD : FWD,
            showGrad && i === 0 && h === 0 ? n2(t.gradW[0][0][0]) : undefined)))}
          {/* edges hidden -> output */}
          {HID.map((hp, h) => edge(hp, OUT, `ho${h}`,
            showGrad ? true : showOut, showGrad ? BWD : FWD,
            showGrad && h === 0 ? n2(t.gradW[1][0][0]) : undefined))}

          {/* nodes */}
          {IN.map((p, i) => <g key={`i${i}`}>{node(p, fwd ? n2(t.a[0][i]) : null, FWD)}</g>)}
          {HID.map((p, h) => <g key={`h${h}`}>{node(p, showHid ? n2(t.a[1][h]) : null, showHidDelta ? BWD : FWD, showHidDelta ? `δ ${n2(t.delta[0][h])}` : null)}</g>)}
          {node(OUT, showOut ? n2(t.output) : null, showOutDelta ? BWD : FWD, showOutDelta ? `δ ${n2(t.delta[1][0])}` : null)}

          <text x={70} y={H - 8} fontSize={9} fill="var(--ink-faint)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">inputs</text>
          <text x={250} y={H - 8} fontSize={9} fill="var(--ink-faint)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">hidden</text>
          <text x={430} y={H - 8} fontSize={9} fill="var(--ink-faint)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">output</text>
          <text x={W - 8} y={20} fontSize={10} fill={stage < 3 ? FWD : BWD} textAnchor="end" fontFamily="var(--font-mono, monospace)">{stage < 3 ? '→ forward' : '← backward'}</text>
        </svg>
      </div>

      {/* worked equation panel */}
      <div className="mt-3 px-4 py-2 border border-rule rounded font-mono text-sm" style={{ color: stage < 3 ? 'var(--ink)' : 'var(--ink)', borderColor: stage < 3 ? 'var(--accent)' : BWD }}>
        {eq}
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => { if (stage >= STAGES.length - 1) setStage(0); setPlaying((p) => !p) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setStage(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <button onClick={() => setStage((s) => Math.min(s + 1, STAGES.length - 1))} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Step</button>
        <input type="range" min={0} max={STAGES.length - 1} value={stage} onChange={(e) => { setPlaying(false); setStage(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">{stage + 1} / {STAGES.length}</span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 22.1 — Backpropagation is the chain rule, flowing backward. First
        the <span style={{ color: FWD }}>forward</span> pass fills every node with
        its value, left to right, and produces the loss. Then the error at the
        output flows <span style={{ color: BWD }}>backward</span> along the very
        same wires: each layer turns the error on its output into the error on its
        input (multiply by the weights and the activation&rsquo;s slope), and
        every weight&rsquo;s gradient is simply its backward error times its
        forward signal. One sweep, every gradient — no million re-runs.
      </figcaption>
    </figure>
  )
}
