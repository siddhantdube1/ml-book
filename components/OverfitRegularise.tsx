'use client'

import { useMemo, useState } from 'react'
import { trainMLP, mlpForward, makeMoonsData, type Point } from '@/lib/neuralnet'

const W = 680
const H = 340
const CLASS = ['#3c5a8c', '#c7522a'] as const
const BX = 8, BY = 30, BS = 290, RANGE = 3.2, G = 40
const LX = 350, LY = 30, LW = 322, LH = 268
const Y_MAX = 0.9
const EPOCHS = 1500

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1, g = 90 + (82 - 90) * p1, b = 140 + (42 - 140) * p1
  return `rgba(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)},${(Math.abs(p1 - 0.5) * 2 * 0.5).toFixed(3)})`
}

export default function OverfitRegularise() {
  const [t, setT] = useState(0) // -> weight decay
  const wd = useMemo(() => t * 0.012, [t])

  const train = useMemo<Point[]>(() => makeMoonsData(36, 7, 0.42), [])
  const val = useMemo<Point[]>(() => makeMoonsData(400, 99, 0.42), [])
  const frames = useMemo(() => trainMLP(train, [20, 20], { epochs: EPOCHS, lr: 0.08, act: 'tanh', seed: 3, weightDecay: wd, valData: val, recordEvery: 25 }), [train, val, wd])

  const last = frames[frames.length - 1]
  const bestVal = useMemo(() => {
    let b = frames[0]
    for (const f of frames) if ((f.valLoss ?? 9) < (b.valLoss ?? 9)) b = f
    return b
  }, [frames])

  const bx = (x: number) => BX + ((x + RANGE) / (2 * RANGE)) * BS
  const by = (y: number) => BY + ((RANGE - y) / (2 * RANGE)) * BS
  const heat = useMemo(() => {
    const cells: { x: number; y: number; s: number; fill: string }[] = []
    const cs = BS / G
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const xD = -RANGE + ((i + 0.5) / G) * 2 * RANGE
      const yD = RANGE - ((j + 0.5) / G) * 2 * RANGE
      cells.push({ x: BX + i * cs, y: BY + j * cs, s: cs, fill: blend(mlpForward(last.net, [xD, yD]).output) })
    }
    return cells
  }, [last])

  const lx = (e: number) => LX + (e / EPOCHS) * LW
  const ly = (l: number) => LY + LH - (Math.min(Math.max(l, 0), Y_MAX) / Y_MAX) * LH
  const pathOf = (key: 'loss' | 'valLoss') => frames.map((f, i) => `${i === 0 ? 'M' : 'L'} ${lx(f.epoch).toFixed(1)} ${ly((f[key] ?? 0)).toFixed(1)}`).join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">weight decay = <span className="font-mono">{wd.toFixed(4)}</span></span><input type="range" min={0} max={1} step={0.0833} value={t} onChange={(e) => setT(parseFloat(e.target.value))} className="w-52" /></label>
        <span className="font-mono text-xs text-ink-muted">val loss: final <span className="text-ink">{last.valLoss?.toFixed(3)}</span> · best <span className="text-accent">{bestVal.valLoss?.toFixed(3)}</span></span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <text x={BX + BS / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">boundary on 36 noisy points</text>
          {heat.map((c, i) => <rect key={i} x={c.x} y={c.y} width={c.s} height={c.s} fill={c.fill} shapeRendering="crispEdges" />)}
          {train.map((p, i) => <circle key={i} cx={bx(p.x[0])} cy={by(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />)}

          {/* loss curves */}
          <text x={LX + LW / 2} y={20} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">training vs validation loss</text>
          <line x1={LX} y1={LY + LH} x2={LX + LW} y2={LY + LH} stroke="var(--rule)" strokeWidth={1} />
          <line x1={LX} y1={LY} x2={LX} y2={LY + LH} stroke="var(--rule)" strokeWidth={1} />
          {/* early-stop marker at validation minimum */}
          <line x1={lx(bestVal.epoch)} y1={LY} x2={lx(bestVal.epoch)} y2={LY + LH} stroke="var(--accent)" strokeWidth={1} strokeDasharray="3,3" opacity={0.6} />
          <text x={lx(bestVal.epoch)} y={LY - 2} fontSize={8.5} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">early stop</text>
          <path d={pathOf('loss')} fill="none" stroke="var(--accent)" strokeWidth={2} />
          <path d={pathOf('valLoss')} fill="none" stroke="#c7522a" strokeWidth={2} />
          {[0, 0.3, 0.6, 0.9].map((v) => <text key={v} x={LX - 5} y={ly(v) + 3} fontSize={9} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{v.toFixed(1)}</text>)}
          <g transform={`translate(${LX + 12}, ${LY + 10})`} fontFamily="var(--font-sans, sans-serif)" fontSize={10}>
            <line x1={0} y1={0} x2={14} y2={0} stroke="var(--accent)" strokeWidth={2} /><text x={19} y={3} fill="var(--ink-muted)">training</text>
            <line x1={0} y1={14} x2={14} y2={14} stroke="#c7522a" strokeWidth={2} /><text x={19} y={17} fill="var(--ink-muted)">validation</text>
          </g>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 22.5 — Networks overfit, and the old cures still work. With no
        weight decay this oversized network drives its <span className="text-accent">training
        loss</span> toward zero by memorising 36 noisy points — but the{' '}
        <span style={{ color: '#c7522a' }}>validation loss</span> bottoms out and
        turns sharply up, the overfitting U of Chapters 11 and 14, now on a
        network, and the boundary contorts around every point. Turn up weight
        decay and the curves pull back together, the boundary smooths, and the
        best validation loss improves. Early stopping (dashed) and dropout do the
        same job — the book&rsquo;s regularisation, on its most powerful model.
      </figcaption>
    </figure>
  )
}
