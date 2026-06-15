'use client'

import { useMemo, useState } from 'react'
import { makeMoons2, accuracy, type Point } from '@/lib/tree'
import { trainForest, forestAccuracy, oobAccuracy } from '@/lib/forest'

const W = 620
const H = 340
const PAD_L = 46
const PAD_R = 18
const PAD_T = 22
const PAD_B = 42
const MAX_B = 100
const Y_MIN = 0.65
const Y_MAX = 1.0

export default function VarianceCurve() {
  const [b, setB] = useState(50)

  const { train, test } = useMemo(
    () => ({
      train: makeMoons2(180, 1, 0.22, 0.06) as Point[],
      test: makeMoons2(180, 99, 0.22, 0.06) as Point[],
    }),
    [],
  )

  const forest = useMemo(
    () =>
      trainForest(train, {
        numTrees: MAX_B,
        numClasses: 2,
        maxDepth: 8,
        minSamplesLeaf: 1,
        maxFeatures: 1,
        seed: 5,
      }),
    [train],
  )

  const { testCurve, oobCurve, singles } = useMemo(() => {
    const testCurve: number[] = []
    const oobCurve: number[] = []
    for (let B = 1; B <= MAX_B; B++) {
      testCurve.push(forestAccuracy(forest, test, B))
      oobCurve.push(oobAccuracy(forest, train, B))
    }
    const singles = forest.trees.map((t) => accuracy(t, test))
    return { testCurve, oobCurve, singles }
  }, [forest, test, train])

  const bx = (B: number) => PAD_L + ((B - 1) / (MAX_B - 1)) * (W - PAD_L - PAD_R)
  const by = (a: number) =>
    H - PAD_B - ((a - Y_MIN) / (Y_MAX - Y_MIN)) * (H - PAD_T - PAD_B)

  const path = (arr: number[]) =>
    arr.map((a, i) => `${i === 0 ? 'M' : 'L'} ${bx(i + 1).toFixed(1)} ${by(a).toFixed(1)}`).join(' ')

  const meanSingle = singles.reduce((a, c) => a + c, 0) / singles.length

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Trees B = <span className="font-mono">{b}</span>
          </span>
          <input type="range" min={1} max={MAX_B} step={1} value={b} onChange={(e) => setB(parseInt(e.target.value))} className="w-48" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line x1={PAD_L} y1={by(Y_MIN)} x2={W - PAD_R} y2={by(Y_MIN)} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={by(Y_MIN)} stroke="var(--rule)" strokeWidth={1} />

          {/* individual-tree accuracies (the variance) */}
          {singles.map((a, i) => (
            <circle key={i} cx={bx(i + 1)} cy={by(Math.max(Y_MIN, a))} r={2} fill="var(--ink-muted)" opacity={0.35} />
          ))}
          {/* mean single-tree line */}
          <line x1={PAD_L} y1={by(meanSingle)} x2={W - PAD_R} y2={by(meanSingle)} stroke="var(--ink-muted)" strokeWidth={1} strokeDasharray="1,3" opacity={0.6} />

          {/* OOB + test curves */}
          <path d={path(oobCurve)} fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.8} />
          <path d={path(testCurve)} fill="none" stroke="var(--accent)" strokeWidth={2} />

          {/* current-B marker */}
          <line x1={bx(b)} y1={PAD_T} x2={bx(b)} y2={by(Y_MIN)} stroke="var(--ink)" strokeWidth={1.25} opacity={0.35} />
          <circle cx={bx(b)} cy={by(testCurve[b - 1])} r={4} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1.25} />

          {/* ticks */}
          {[0.7, 0.8, 0.9, 1.0].map((t) => (
            <text key={t} x={PAD_L - 6} y={by(t) + 3} fontSize={10} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{t.toFixed(2)}</text>
          ))}
          {[1, 25, 50, 75, 100].map((B) => (
            <text key={B} x={bx(B)} y={by(Y_MIN) + 16} fontSize={10} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{B}</text>
          ))}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 6} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">number of trees B</text>
          <text x={PAD_L} y={PAD_T - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">accuracy</text>

          {/* legend */}
          <g transform={`translate(${W - PAD_R - 150}, ${by(Y_MIN) - 60})`}>
            <line x1={0} y1={0} x2={16} y2={0} stroke="var(--accent)" strokeWidth={2} />
            <text x={20} y={3.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">forest test accuracy</text>
            <line x1={0} y1={14} x2={16} y2={14} stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="4,3" />
            <text x={20} y={17.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">out-of-bag accuracy</text>
            <circle cx={8} cy={28} r={2} fill="var(--ink-muted)" opacity={0.5} />
            <text x={20} y={31.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">single trees</text>
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>forest @ B={b} = <span className="text-accent">{testCurve[b - 1].toFixed(3)}</span></div>
        <div>OOB @ B={b} = <span className="text-ink">{oobCurve[b - 1].toFixed(3)}</span></div>
        <div>mean single tree = <span className="text-ink">{meanSingle.toFixed(3)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 13.3 — What the forest buys. Each faint dot is one tree's own
        test accuracy — scattered, and mostly below the forest. The forest's
        accuracy (solid teal) climbs as trees are added and then plateaus:
        more trees only ever refine, never overfit. The out-of-bag accuracy
        (dashed) tracks the held-out test accuracy closely — a validation
        estimate the forest computes for free, no separate split needed.
      </figcaption>
    </figure>
  )
}
