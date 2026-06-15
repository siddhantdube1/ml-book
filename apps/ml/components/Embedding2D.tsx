'use client'

import { useMemo, useState } from 'react'
import { highDimClusters, pca, project } from '@/lib/pca'

const W = 680
const H = 360
const DIM = 8
const PALETTE = ['#1d6d5e', '#c7522a', '#3c5a8c']

const PAIRS: [number, number][] = [
  [0, 1],
  [2, 3],
  [4, 5],
  [6, 7],
]

function fitScatter(coords: number[][], ox: number, oy: number, pw: number, ph: number) {
  const xs = coords.map((c) => c[0])
  const ys = coords.map((c) => c[1])
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const sx = (x: number) => ox + 16 + ((x - x0) / (x1 - x0 || 1)) * (pw - 32)
  const sy = (y: number) => oy + ph - 16 - ((y - y0) / (y1 - y0 || 1)) * (ph - 32)
  return { sx, sy }
}

export default function Embedding2D() {
  const [pairIdx, setPairIdx] = useState(0)

  const { X, y } = useMemo(() => highDimClusters(240, DIM, 3, 3), [])
  const model = useMemo(() => pca(X, true), [X])
  const scores = useMemo(() => project(model, X, 2), [model, X])

  const pair = PAIRS[pairIdx]
  const rawCoords = useMemo(() => X.map((r) => [r[pair[0]], r[pair[1]]]), [X, pair])

  const PW = 320
  const PH = 280
  const left = fitScatter(rawCoords, 6, 40, PW, PH)
  const right = fitScatter(scores, 354, 40, PW, PH)

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-4 gap-y-3 items-center mb-4 font-sans text-sm">
        <span className="text-ink-muted">Raw feature pair:</span>
        <div className="flex flex-wrap gap-1.5">
          {PAIRS.map((p, i) => (
            <button key={i} onClick={() => setPairIdx(i)} className="px-2.5 py-1 border rounded font-mono text-xs transition-colors" style={{ borderColor: pairIdx === i ? 'var(--accent)' : 'var(--rule)', background: pairIdx === i ? 'var(--accent)' : 'transparent', color: pairIdx === i ? 'var(--paper)' : 'var(--ink-muted)' }}>
              x{p[0] + 1},x{p[1] + 1}
            </button>
          ))}
        </div>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <rect x={6} y={40} width={PW} height={PH} fill="none" stroke="var(--rule)" strokeWidth={0.75} rx={4} />
          <rect x={354} y={40} width={PW} height={PH} fill="none" stroke="var(--rule)" strokeWidth={0.75} rx={4} />
          <text x={6 + PW / 2} y={28} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">
            raw features x{pair[0] + 1}, x{pair[1] + 1} (of {DIM})
          </text>
          <text x={354 + PW / 2} y={28} fontSize={11} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">
            projected onto PC1 × PC2
          </text>

          {rawCoords.map((c, i) => (
            <circle key={`l${i}`} cx={left.sx(c[0])} cy={left.sy(c[1])} r={2.8} fill={PALETTE[y[i]]} stroke="var(--paper)" strokeWidth={0.4} opacity={0.85} />
          ))}
          {scores.map((c, i) => (
            <circle key={`r${i}`} cx={right.sx(c[0])} cy={right.sy(c[1])} r={2.8} fill={PALETTE[y[i]]} stroke="var(--paper)" strokeWidth={0.4} opacity={0.85} />
          ))}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 20.4 — Seeing high-dimensional data. There are three clusters
        hidden in these eight features, separated along directions spread across
        all of them. Look at any pair of raw features (left) and the clusters
        smear together — no two original axes reveal the structure. Project the
        same points onto the top two principal components (right) and the three
        clusters fall cleanly apart. Cycle the raw pairs: none of them match what
        PCA finds in a single, automatic rotation. This is why PCA is the first
        thing you reach for to look at unfamiliar data.
      </figcaption>
    </figure>
  )
}
