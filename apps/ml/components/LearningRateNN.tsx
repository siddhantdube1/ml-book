'use client'

import { useEffect, useMemo, useState } from 'react'
import { trainMLP, mlpForward, makeMoonsData, type Point } from '@/lib/neuralnet'

const W = 680
const H = 340
const CLASS = ['#3c5a8c', '#c7522a'] as const
// left boundary panel
const BX = 8, BY = 30, BS = 290, RANGE = 3.2, G = 40
// right loss panel
const LX = 350, LY = 30, LW = 322, LH = 268
const LR_MIN = 0.01, LR_MAX = 50
const Y_MAX = 1.4

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1, g = 90 + (82 - 90) * p1, b = 140 + (42 - 140) * p1
  return `rgba(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)},${(Math.abs(p1 - 0.5) * 2 * 0.5).toFixed(3)})`
}

export default function LearningRateNN() {
  const [t, setT] = useState(0.55) // -> log lr ~ 0.5
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const lr = useMemo(() => Math.pow(10, Math.log10(LR_MIN) + t * (Math.log10(LR_MAX) - Math.log10(LR_MIN))), [t])
  const data = useMemo<Point[]>(() => makeMoonsData(160, 5), [])
  const frames = useMemo(() => trainMLP(data, [12, 12], { epochs: 160, lr, act: 'tanh', seed: 2 }), [data, lr])

  useEffect(() => { setIdx(0); setPlaying(false) }, [frames])
  useEffect(() => {
    if (!playing) return
    if (idx >= frames.length - 1) { setPlaying(false); return }
    const id = setTimeout(() => setIdx((i) => i + 1), 90)
    return () => clearTimeout(id)
  }, [playing, idx, frames.length])

  const cur = frames[Math.min(idx, frames.length - 1)]
  const bx = (x: number) => BX + ((x + RANGE) / (2 * RANGE)) * BS
  const by = (y: number) => BY + ((RANGE - y) / (2 * RANGE)) * BS

  const heat = useMemo(() => {
    const cells: { x: number; y: number; s: number; fill: string }[] = []
    const cs = BS / G
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const xD = -RANGE + ((i + 0.5) / G) * 2 * RANGE
      const yD = RANGE - ((j + 0.5) / G) * 2 * RANGE
      cells.push({ x: BX + i * cs, y: BY + j * cs, s: cs, fill: blend(mlpForward(cur.net, [xD, yD]).output) })
    }
    return cells
  }, [cur])

  const maxEpoch = frames[frames.length - 1].epoch
  const lx = (e: number) => LX + (e / maxEpoch) * LW
  const ly = (l: number) => LY + LH - (Math.min(Math.max(l, 0), Y_MAX) / Y_MAX) * LH
  const path = (upto: number) => frames.filter((f) => f.epoch <= upto).map((f, i) => `${i === 0 ? 'M' : 'L'} ${lx(f.epoch).toFixed(1)} ${ly(f.loss).toFixed(1)}`).join(' ')

  const regime = lr < 0.08 ? 'too small — crawls' : lr > 8 ? 'too large — unstable' : 'good'

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">learning rate = <span className="font-mono">{lr < 1 ? lr.toFixed(3) : lr.toFixed(1)}</span></span><input type="range" min={0} max={1} step={0.01} value={t} onChange={(e) => setT(parseFloat(e.target.value))} className="w-52" /></label>
        <span className="font-mono text-xs" style={{ color: regime === 'good' ? 'var(--accent)' : '#c7522a' }}>{regime}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <text x={BX + BS / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">decision boundary</text>
          {heat.map((c, i) => <rect key={i} x={c.x} y={c.y} width={c.s} height={c.s} fill={c.fill} shapeRendering="crispEdges" />)}
          {data.map((p, i) => <circle key={i} cx={bx(p.x[0])} cy={by(p.x[1])} r={2.6} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.5} />)}

          {/* loss curve */}
          <text x={LX + LW / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">training loss vs epoch</text>
          <line x1={LX} y1={LY + LH} x2={LX + LW} y2={LY + LH} stroke="var(--rule)" strokeWidth={1} />
          <line x1={LX} y1={LY} x2={LX} y2={LY + LH} stroke="var(--rule)" strokeWidth={1} />
          <path d={frames.map((f, i) => `${i === 0 ? 'M' : 'L'} ${lx(f.epoch).toFixed(1)} ${ly(f.loss).toFixed(1)}`).join(' ')} fill="none" stroke="var(--ink-faint)" strokeWidth={1} opacity={0.5} />
          <path d={path(cur.epoch)} fill="none" stroke="var(--accent)" strokeWidth={2} />
          <circle cx={lx(cur.epoch)} cy={ly(cur.loss)} r={3.5} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1} />
          {[0, 0.7, 1.4].map((v) => <text key={v} x={LX - 5} y={ly(v) + 3} fontSize={9} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{v.toFixed(1)}</text>)}
          <text x={LX + LW} y={LY + LH + 14} fontSize={9} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">epoch {maxEpoch}</text>
        </svg>
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => { if (idx >= frames.length - 1) setIdx(0); setPlaying((p) => !p) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setIdx(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <input type="range" min={0} max={frames.length - 1} value={idx} onChange={(e) => { setPlaying(false); setIdx(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">epoch {cur.epoch} · loss {cur.loss.toFixed(3)}</span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 22.2 — The learning rate decides everything. Press Play and watch
        the loss fall and the boundary form. Set the rate too small and the loss
        barely moves — training crawls and never arrives. Set it too large and the
        loss spikes and thrashes, the boundary lurching wildly as each step
        overshoots. In the sweet spot between, the loss glides smoothly down and
        the moons separate cleanly. It is the same knob as Chapter 6, now steering
        a whole network.
      </figcaption>
    </figure>
  )
}
