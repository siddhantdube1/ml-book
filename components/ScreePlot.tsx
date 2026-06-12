'use client'

import { useMemo, useState } from 'react'
import { decayingGaussian, pca } from '@/lib/pca'

const W = 560
const H = 360
const PAD_L = 48
const PAD_R = 44
const PAD_T = 24
const PAD_B = 44
const DIM = 8

export default function ScreePlot() {
  const [k, setK] = useState(3)
  const data = useMemo(() => decayingGaussian(300, DIM, 5), [])
  const model = useMemo(() => pca(data), [data])

  const cum = useMemo(() => {
    const out: number[] = []
    let s = 0
    for (const r of model.ratios) { s += r; out.push(s) }
    return out
  }, [model])

  const bw = (W - PAD_L - PAD_R) / DIM
  const bx = (i: number) => PAD_L + i * bw + bw * 0.5
  const byBar = (v: number) => H - PAD_B - v * (H - PAD_T - PAD_B)
  const byCum = (v: number) => H - PAD_B - v * (H - PAD_T - PAD_B)

  const cumPath = cum.map((c, i) => `${i === 0 ? 'M' : 'L'} ${bx(i).toFixed(1)} ${byCum(c).toFixed(1)}`).join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Keep <span className="font-mono">{k}</span> of {DIM} components</span>
          <input type="range" min={1} max={DIM} step={1} value={k} onChange={(e) => setK(parseInt(e.target.value))} className="w-44" />
        </label>
        <span className="font-mono text-xs text-accent">{(cum[k - 1] * 100).toFixed(1)}% of variance retained</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />
          {/* 90% guide */}
          <line x1={PAD_L} y1={byCum(0.9)} x2={W - PAD_R} y2={byCum(0.9)} stroke="var(--ink-faint)" strokeWidth={1} strokeDasharray="3,3" />
          <text x={W - PAD_R + 4} y={byCum(0.9) + 3} fontSize={9} fill="var(--ink-faint)" fontFamily="var(--font-mono, monospace)">90%</text>

          {/* eigenvalue bars */}
          {model.ratios.map((r, i) => (
            <rect key={i} x={PAD_L + i * bw + bw * 0.18} y={byBar(r)} width={bw * 0.64} height={H - PAD_B - byBar(r)} fill={i < k ? 'var(--accent)' : 'var(--ink-faint)'} opacity={i < k ? 0.85 : 0.5} />
          ))}

          {/* cumulative curve */}
          <path d={cumPath} fill="none" stroke="var(--ink)" strokeWidth={1.75} />
          {cum.map((c, i) => (
            <circle key={i} cx={bx(i)} cy={byCum(c)} r={i === k - 1 ? 4 : 2.5} fill={i < k ? 'var(--ink)' : 'var(--ink-faint)'} />
          ))}

          {/* labels */}
          {model.ratios.map((_, i) => (
            <text key={i} x={bx(i)} y={H - PAD_B + 15} fontSize={9} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{i + 1}</text>
          ))}
          {[0, 0.5, 1].map((v) => (
            <text key={v} x={PAD_L - 6} y={byBar(v) + 3} fontSize={9} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{(v * 100).toFixed(0)}</text>
          ))}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 8} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">principal component</text>
          {/* legend */}
          <g transform={`translate(${W - PAD_R - 168}, ${PAD_T + 2})`} fontFamily="var(--font-sans, sans-serif)" fontSize={10}>
            <rect x={0} y={-6} width={12} height={8} fill="var(--accent)" opacity={0.85} /><text x={18} y={1.5} fill="var(--ink-muted)">variance per PC</text>
            <line x1={0} y1={14} x2={12} y2={14} stroke="var(--ink)" strokeWidth={1.75} /><text x={18} y={17.5} fill="var(--ink-muted)">cumulative</text>
          </g>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 20.3 — How many components? Each bar is one component&rsquo;s share
        of the total variance — its eigenvalue — and they fall away steeply: the
        first few directions carry almost everything, the rest are scree. The
        black curve adds them up, climbing fast then flattening. Slide{' '}
        <span className="font-mono">k</span> and read where the cumulative line
        crosses the 90% guide — here just three of eight components capture the
        data, so five can be discarded for almost no loss.
      </figcaption>
    </figure>
  )
}
