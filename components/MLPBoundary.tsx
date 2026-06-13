'use client'

import { useEffect, useMemo, useState } from 'react'
import { trainMLP, mlpForward, makeXORData, makeMoonsData, makeCirclesData, type Point } from '@/lib/neuralnet'

const W = 420
const H = 400
const PAD = 22
const RANGE = 3.2
const GX = 50
const GY = 48
const CLASS = ['#3c5a8c', '#c7522a'] as const

const DATASETS = ['XOR', 'moons', 'circles'] as const
type DS = (typeof DATASETS)[number]
const EPOCHS: Record<DS, number> = { XOR: 400, moons: 600, circles: 600 }

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1, g = 90 + (82 - 90) * p1, b = 140 + (42 - 140) * p1
  return `rgba(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)},${(Math.abs(p1 - 0.5) * 2 * 0.5).toFixed(3)})`
}

export default function MLPBoundary() {
  const [ds, setDs] = useState<DS>('XOR')
  const [hidden, setHidden] = useState(4)
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const data = useMemo<Point[]>(() => (ds === 'XOR' ? makeXORData(150, 3) : ds === 'moons' ? makeMoonsData(150, 5) : makeCirclesData(150, 5)), [ds])
  const frames = useMemo(() => trainMLP(data, [hidden], { epochs: EPOCHS[ds], lr: 0.5, act: 'tanh', seed: 2 }), [data, hidden, ds])

  useEffect(() => { setIdx(0); setPlaying(false) }, [frames])
  useEffect(() => {
    if (!playing) return
    if (idx >= frames.length - 1) { setPlaying(false); return }
    const t = setTimeout(() => setIdx((i) => i + 1), 110)
    return () => clearTimeout(t)
  }, [playing, idx, frames.length])

  const cur = frames[Math.min(idx, frames.length - 1)]
  const sx = (x: number) => PAD + ((x + RANGE) / (2 * RANGE)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + RANGE) / (2 * RANGE)) * (H - 2 * PAD)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (W - 2 * PAD) / GX
    const ch = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
      const xD = -RANGE + ((i + 0.5) / GX) * 2 * RANGE
      const yD = RANGE - ((j + 0.5) / GY) * 2 * RANGE
      cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(mlpForward(cur.net, [xD, yD]).output) })
    }
    return cells
  }, [cur])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <div className="flex rounded border border-rule overflow-hidden">
          {DATASETS.map((d) => (
            <button key={d} onClick={() => setDs(d)} className="px-3 py-1 transition-colors" style={{ background: ds === d ? 'var(--accent)' : 'transparent', color: ds === d ? 'var(--paper)' : 'var(--ink-muted)' }}>{d}</button>
          ))}
        </div>
        <label className="flex items-center gap-2"><span className="text-ink-muted">hidden units = <span className="font-mono">{hidden}</span></span><input type="range" min={1} max={10} step={1} value={hidden} onChange={(e) => setHidden(parseInt(e.target.value))} className="w-40" /></label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          {/* hidden-unit lines */}
          {cur.net.W[0].map((row, i) => {
            if (Math.abs(row[1]) < 1e-6) return null
            const ly = (x: number) => -(row[0] * x + cur.net.b[0][i]) / row[1]
            return <line key={i} x1={sx(-RANGE)} y1={sy(ly(-RANGE))} x2={sx(RANGE)} y2={sy(ly(RANGE))} stroke="var(--ink-muted)" strokeWidth={0.75} opacity={0.4} />
          })}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />
          ))}
        </svg>
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => { if (idx >= frames.length - 1) setIdx(0); setPlaying((p) => !p) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setIdx(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <input type="range" min={0} max={frames.length - 1} value={idx} onChange={(e) => { setPlaying(false); setIdx(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">epoch {cur.epoch} · acc {(cur.accuracy * 100).toFixed(0)}%</span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 21.4 — A network bends the boundary. Press Play and watch the
        decision surface form as the network trains. Each faint line is one
        hidden unit&rsquo;s own boundary; the output neuron blends them, and where
        they overlap the surface folds. With a single hidden unit the boundary is
        straight and XOR is hopeless — add units and it wraps the checkerboard,
        the moons, the rings. This is the kernel trick and the basis expansion of
        earlier chapters, except the network builds the folds itself.
      </figcaption>
    </figure>
  )
}
