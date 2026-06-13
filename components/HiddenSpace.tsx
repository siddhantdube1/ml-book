'use client'

import { useEffect, useMemo, useState } from 'react'
import { trainMLP, hiddenActivations, makeXORData, makeMoonsData, type Point } from '@/lib/neuralnet'

const W = 680
const H = 360
const CLASS = ['#3c5a8c', '#c7522a'] as const

export default function HiddenSpace() {
  const [ds, setDs] = useState<'XOR' | 'moons'>('XOR')
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const data = useMemo<Point[]>(() => (ds === 'XOR' ? makeXORData(120, 3) : makeMoonsData(120, 5)), [ds])
  const frames = useMemo(() => trainMLP(data, [2], { epochs: ds === 'XOR' ? 800 : 900, lr: 0.7, act: 'tanh', seed: 7 }), [data, ds])

  useEffect(() => { setIdx(0); setPlaying(false) }, [frames])
  useEffect(() => {
    if (!playing) return
    if (idx >= frames.length - 1) { setPlaying(false); return }
    const t = setTimeout(() => setIdx((i) => i + 1), 90)
    return () => clearTimeout(t)
  }, [playing, idx, frames.length])

  const cur = frames[Math.min(idx, frames.length - 1)]

  // left: input space [-3,3]
  const L0 = 14, LW = 312, IR = 3.2
  const lsx = (x: number) => L0 + 12 + ((x + IR) / (2 * IR)) * (LW - 24)
  const lsy = (y: number) => H - 24 - ((y + IR) / (2 * IR)) * (H - 60)
  // right: hidden space [-1.1,1.1]
  const R0 = 360, RW = 306, HR = 1.12
  const rsx = (h: number) => R0 + 12 + ((h + HR) / (2 * HR)) * (RW - 24)
  const rsy = (h: number) => H - 24 - ((h + HR) / (2 * HR)) * (H - 60)

  const hidden = useMemo(() => data.map((p) => hiddenActivations(cur.net, p.x, 0)), [cur, data])

  // output neuron line in hidden space: w0*h1 + w1*h2 + b = 0
  const ow = cur.net.W[1][0]
  const ob = cur.net.b[1][0]
  const outLineY = (h1: number) => -(ow[0] * h1 + ob) / ow[1]

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <div className="flex rounded border border-rule overflow-hidden">
          {(['XOR', 'moons'] as const).map((d) => (
            <button key={d} onClick={() => setDs(d)} className="px-3 py-1 transition-colors" style={{ background: ds === d ? 'var(--accent)' : 'transparent', color: ds === d ? 'var(--paper)' : 'var(--ink-muted)' }}>{d}</button>
          ))}
        </div>
        <span className="font-mono text-xs text-ink-muted">2 hidden units · acc {(cur.accuracy * 100).toFixed(0)}%</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <text x={L0 + LW / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">input space (not separable)</text>
          <text x={R0 + RW / 2} y={20} fontSize={11} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">hidden-unit space (h₁, h₂)</text>
          <line x1={342} y1={28} x2={342} y2={H - 14} stroke="var(--rule)" strokeWidth={1} />

          {/* left: input space + hidden-unit lines */}
          {cur.net.W[0].map((row, i) => {
            if (Math.abs(row[1]) < 1e-6) return null
            const ly = (x: number) => -(row[0] * x + cur.net.b[0][i]) / row[1]
            return <line key={i} x1={lsx(-IR)} y1={lsy(ly(-IR))} x2={lsx(IR)} y2={lsy(ly(IR))} stroke="var(--ink-muted)" strokeWidth={0.9} opacity={0.5} strokeDasharray="4,3" />
          })}
          {data.map((p, i) => (
            <circle key={`l${i}`} cx={lsx(p.x[0])} cy={lsy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />
          ))}

          {/* right: hidden space + output line */}
          <rect x={R0 + 12} y={H - 24 - (H - 60)} width={RW - 24} height={H - 60} fill="none" stroke="var(--rule)" strokeWidth={0.5} />
          {Math.abs(ow[1]) > 1e-6 && (
            <line x1={rsx(-HR)} y1={rsy(outLineY(-HR))} x2={rsx(HR)} y2={rsy(outLineY(HR))} stroke="var(--accent)" strokeWidth={2} />
          )}
          {hidden.map((h, i) => (
            <circle key={`r${i}`} cx={rsx(h[0])} cy={rsy(h[1])} r={3} fill={CLASS[data[i].y]} stroke="var(--paper)" strokeWidth={0.6} />
          ))}
        </svg>
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => { if (idx >= frames.length - 1) setIdx(0); setPlaying((p) => !p) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setIdx(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <input type="range" min={0} max={frames.length - 1} value={idx} onChange={(e) => { setPlaying(false); setIdx(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">epoch {cur.epoch}</span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 21.5 — A hidden layer learns a new representation. On the left, the
        classes are tangled — no straight line separates them. The two hidden
        units (dashed) each measure one direction, and on the right the same
        points are re-plotted in <em>their</em> coordinates (h₁, h₂). Press Play:
        as the network trains, it warps that hidden space until the classes pull
        cleanly apart, and a single straight line — the output neuron (teal) —
        suffices. The network learned the very lift the kernel trick (Chapter 15)
        and basis expansion (Chapter 17) had to be handed.
      </figcaption>
    </figure>
  )
}
