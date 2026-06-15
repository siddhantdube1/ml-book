'use client'

import { useEffect, useMemo, useState } from 'react'
import { trainMLP, makeMoonsData, type Optimizer } from '@/lib/neuralnet'

const W = 600
const H = 340
const PAD_L = 46, PAD_R = 16, PAD_T = 24, PAD_B = 40
const EPOCHS = 250
const Y_MAX = 1.2

const OPTS: { key: Optimizer; label: string; colour: string; lr: number }[] = [
  { key: 'sgd', label: 'plain SGD', colour: '#3c5a8c', lr: 0.5 },
  { key: 'momentum', label: 'momentum', colour: '#c7522a', lr: 0.5 },
  { key: 'adam', label: 'Adam', colour: '#1d6d5e', lr: 0.05 },
]

export default function OptimizerRace() {
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(true)

  const data = useMemo(() => makeMoonsData(160, 5), [])
  const runs = useMemo(
    () => OPTS.map((o) => ({ ...o, frames: trainMLP(data, [8], { epochs: EPOCHS, lr: o.lr, act: 'tanh', seed: 2, optimizer: o.key, recordEvery: 4 }) })),
    [data],
  )
  const nFrames = runs[0].frames.length

  useEffect(() => {
    if (!playing) return
    if (idx >= nFrames - 1) { setPlaying(false); return }
    const id = setTimeout(() => setIdx((i) => i + 1), 70)
    return () => clearTimeout(id)
  }, [playing, idx, nFrames])

  const lx = (e: number) => PAD_L + (e / EPOCHS) * (W - PAD_L - PAD_R)
  const ly = (l: number) => H - PAD_B - (Math.min(l, Y_MAX) / Y_MAX) * (H - PAD_T - PAD_B)
  const curEpoch = runs[0].frames[Math.min(idx, nFrames - 1)].epoch

  return (
    <figure className="my-10">
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />
          {[0, 0.4, 0.8, 1.2].map((v) => <text key={v} x={PAD_L - 6} y={ly(v) + 3} fontSize={9} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{v.toFixed(1)}</text>)}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 8} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">epoch</text>
          <text x={PAD_L} y={PAD_T - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">training loss</text>

          {runs.map((run) => {
            const visible = run.frames.filter((f) => f.epoch <= curEpoch)
            const d = visible.map((f, i) => `${i === 0 ? 'M' : 'L'} ${lx(f.epoch).toFixed(1)} ${ly(f.loss).toFixed(1)}`).join(' ')
            const last = visible[visible.length - 1]
            return (
              <g key={run.key}>
                <path d={d} fill="none" stroke={run.colour} strokeWidth={2} />
                {last && <circle cx={lx(last.epoch)} cy={ly(last.loss)} r={3.5} fill={run.colour} stroke="var(--paper)" strokeWidth={1} />}
              </g>
            )
          })}

          {/* legend */}
          <g transform={`translate(${W - PAD_R - 150}, ${PAD_T + 6})`} fontFamily="var(--font-sans, sans-serif)" fontSize={11}>
            {runs.map((run, i) => (
              <g key={run.key} transform={`translate(0, ${i * 18})`}>
                <line x1={0} y1={0} x2={16} y2={0} stroke={run.colour} strokeWidth={2.5} />
                <text x={22} y={3.5} fill="var(--ink-muted)">{run.label} <tspan fill={run.colour}>({run.frames[Math.min(idx, nFrames - 1)].loss.toFixed(3)})</tspan></text>
              </g>
            ))}
          </g>
        </svg>
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => { if (idx >= nFrames - 1) setIdx(0); setPlaying((p) => !p) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setIdx(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <input type="range" min={0} max={nFrames - 1} value={idx} onChange={(e) => { setPlaying(false); setIdx(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">epoch {curEpoch}</span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 22.4 — The optimiser matters. All three descend the same network on
        the same data, differing only in how they use the gradient.{' '}
        <span style={{ color: '#3c5a8c' }}>Plain SGD</span> inches down and stalls
        on the flats. <span style={{ color: '#c7522a' }}>Momentum</span>{' '}
        accumulates speed and rolls through them.{' '}
        <span style={{ color: '#1d6d5e' }}>Adam</span> gives every weight its own
        adaptive step and reaches a low loss in a fraction of the epochs — which
        is why it is the default optimiser of modern deep learning.
      </figcaption>
    </figure>
  )
}
