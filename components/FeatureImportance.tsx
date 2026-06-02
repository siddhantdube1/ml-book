'use client'

import { useMemo, useState } from 'react'
import { makeSparseDataset } from '@/lib/regularisation'
import { trainForest, featureImportances } from '@/lib/forest'
import type { Point } from '@/lib/tree'

const W = 600
const H = 320
const PAD_L = 92
const PAD_R = 30
const PAD_T = 24
const PAD_B = 36

const N_SIGNAL = 3
const N_NOISE = 5
const P = N_SIGNAL + N_NOISE
const MAX_TREES = 150

export default function FeatureImportance() {
  const [nTrees, setNTrees] = useState(80)

  const data = useMemo<Point[]>(
    () =>
      makeSparseDataset(300, N_SIGNAL, N_NOISE, [2.0, -1.5, 1.0], 7, 0.4).map(
        (d) => ({ x: d.x, y: d.y }),
      ),
    [],
  )

  // Train the full forest once; the slider re-aggregates the first nTrees.
  const forest = useMemo(
    () =>
      trainForest(data, {
        numTrees: MAX_TREES,
        numClasses: 2,
        maxDepth: 8,
        minSamplesLeaf: 2,
        maxFeatures: 3,
        seed: 11,
      }),
    [data],
  )

  const imp = useMemo(
    () => featureImportances(forest, P, nTrees),
    [forest, nTrees],
  )

  const maxImp = Math.max(...imp, 0.01)
  const rowH = (H - PAD_T - PAD_B) / P
  const bx = (v: number) => PAD_L + (v / (maxImp * 1.08)) * (W - PAD_L - PAD_R)

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Trees = <span className="font-mono">{nTrees}</span>
          </span>
          <input
            type="range"
            min={1}
            max={MAX_TREES}
            step={1}
            value={nTrees}
            onChange={(e) => setNTrees(parseInt(e.target.value))}
            className="w-48"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axis */}
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--rule)" strokeWidth={1} />

          {imp.map((v, f) => {
            const y = PAD_T + f * rowH
            const signal = f < N_SIGNAL
            return (
              <g key={f}>
                <text x={PAD_L - 8} y={y + rowH / 2 + 4} fontSize={11} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">
                  x{f + 1} {signal ? '(signal)' : '(noise)'}
                </text>
                <rect
                  x={PAD_L + 1}
                  y={y + rowH * 0.18}
                  width={Math.max(0, bx(v) - PAD_L)}
                  height={rowH * 0.64}
                  fill={signal ? 'var(--accent)' : 'var(--ink-faint)'}
                  opacity={signal ? 0.85 : 0.55}
                  rx={2}
                />
                <text x={bx(v) + 6} y={y + rowH / 2 + 4} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-mono, monospace)">
                  {v.toFixed(3)}
                </text>
              </g>
            )
          })}

          <text x={(PAD_L + W - PAD_R) / 2} y={H - 6} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">
            feature importance (share of total impurity decrease)
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          signal share ={' '}
          <span className="text-accent">
            {(imp.slice(0, N_SIGNAL).reduce((a, b) => a + b, 0) * 100).toFixed(0)}%
          </span>
        </div>
        <div>
          noise share ={' '}
          <span className="text-ink">
            {(imp.slice(N_SIGNAL).reduce((a, b) => a + b, 0) * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 13.4 — Feature importance from a forest trained on data with
        three real signal features (x₁–x₃) and five pure-noise features
        (x₄–x₈). Each bar is the share of total impurity decrease the forest
        credits to that feature. The signal features tower over the noise —
        the forest has recovered which inputs matter, the same diagnosis the
        lasso gave in Chapter 9, by an entirely different mechanism. Raise the
        number of trees and the ranking settles down.
      </figcaption>
    </figure>
  )
}
