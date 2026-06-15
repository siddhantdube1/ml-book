'use client'

import { useEffect, useMemo, useState } from 'react'
import { createRng, gauss } from '@/lib/rng'
import { trainPerceptron, makeXORData, type Point } from '@/lib/neuralnet'

const W = 400
const H = 400
const PAD = 26
const RANGE = 3.2
const CLASS = ['#3c5a8c', '#c7522a'] as const

function separable(seed: number): Point[] {
  const rng = createRng(seed)
  const out: Point[] = []
  for (let i = 0; i < 50; i++) {
    const y = (i % 2) as 0 | 1
    out.push({ x: [gauss(rng, y ? 1.5 : -1.5, 0.55), gauss(rng, y ? 0.7 : -0.7, 0.55)], y })
  }
  return out
}

export default function PerceptronLearning() {
  const [dataset, setDataset] = useState<'separable' | 'xor'>('separable')
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const data = useMemo<Point[]>(() => (dataset === 'separable' ? separable(4) : makeXORData(60, 3)), [dataset])
  const frames = useMemo(() => trainPerceptron(data, 60, 0.1), [data])

  useEffect(() => { setIdx(0); setPlaying(false) }, [frames])
  useEffect(() => {
    if (!playing) return
    if (idx >= frames.length - 1) { setPlaying(false); return }
    const t = setTimeout(() => setIdx((i) => i + 1), dataset === 'xor' ? 90 : 360)
    return () => clearTimeout(t)
  }, [playing, idx, frames.length, dataset])

  const sx = (x: number) => PAD + ((x + RANGE) / (2 * RANGE)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + RANGE) / (2 * RANGE)) * (H - 2 * PAD)

  const cur = frames[Math.min(idx, frames.length - 1)]
  const lineY = (x: number) => -(cur.w[0] * x + cur.b) / cur.w[1]
  const converged = dataset === 'separable' && frames[frames.length - 1].errors === 0

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <div className="flex rounded border border-rule overflow-hidden">
          {(['separable', 'xor'] as const).map((d) => (
            <button key={d} onClick={() => setDataset(d)} className="px-3 py-1 transition-colors" style={{ background: dataset === d ? 'var(--accent)' : 'transparent', color: dataset === d ? 'var(--paper)' : 'var(--ink-muted)' }}>{d === 'separable' ? 'separable' : 'XOR'}</button>
          ))}
        </div>
        <span className="font-mono text-xs text-ink-muted">errors = <span style={{ color: cur.errors === 0 ? 'var(--accent)' : 'var(--ink)' }}>{cur.errors}</span> / {data.length}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {Math.abs(cur.w[1]) > 1e-6 && (
            <line x1={sx(-RANGE)} y1={sy(lineY(-RANGE))} x2={sx(RANGE)} y2={sy(lineY(RANGE))} stroke="var(--accent)" strokeWidth={2} />
          )}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={cur.picked === i ? 6 : 3.2} fill={CLASS[p.y]} stroke={cur.picked === i ? 'var(--ink)' : 'var(--paper)'} strokeWidth={cur.picked === i ? 1.6 : 0.7} />
          ))}
          {converged && idx >= frames.length - 1 && (
            <text x={W / 2} y={PAD + 4} fontSize={12} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">converged — every point correct</text>
          )}
          {dataset === 'xor' && idx >= frames.length - 1 && (
            <text x={W / 2} y={PAD + 4} fontSize={12} fill="#c7522a" textAnchor="middle" fontFamily="var(--font-mono, monospace)">no line exists — never converges</text>
          )}
        </svg>
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => { if (idx >= frames.length - 1) setIdx(0); setPlaying((p) => !p) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setIdx(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <button onClick={() => setIdx((i) => Math.min(i + 1, frames.length - 1))} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Step</button>
        <input type="range" min={0} max={frames.length - 1} value={idx} onChange={(e) => { setPlaying(false); setIdx(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">update {idx} / {frames.length - 1}</span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 21.2 — The perceptron learning rule. Each update picks a
        misclassified point (ringed) and nudges the weights toward getting it
        right, rotating the boundary. On <span className="font-mono">separable</span>
        data this provably converges to a line that splits the classes, then
        stops. Switch to <span className="font-mono">XOR</span> and the same loop
        thrashes forever — lurching the line back and forth, never settling —
        because no straight line can separate the checkerboard. One neuron has
        hit its wall.
      </figcaption>
    </figure>
  )
}
