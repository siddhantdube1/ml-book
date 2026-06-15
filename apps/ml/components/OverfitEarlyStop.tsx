'use client'

import { useMemo, useState } from 'react'
import { makeNoisyScatter } from '@/lib/regularisation'
import { trainGBRegressor, gbPredict } from '@/lib/boosting'

const W = 620
const H = 340
const PAD_L = 50
const PAD_R = 18
const PAD_T = 22
const PAD_B = 42
const ROUNDS = 120
const DEPTH = 2
const LR_MIN = 0.05
const LR_MAX = 0.5

export default function OverfitEarlyStop() {
  const [t, setT] = useState(0.34) // -> log lr, default ~0.1

  const lr = useMemo(() => {
    const lo = Math.log10(LR_MIN)
    const hi = Math.log10(LR_MAX)
    return Math.pow(10, lo + t * (hi - lo))
  }, [t])

  const { train, test } = useMemo(
    () => ({ train: makeNoisyScatter(70, 3, 0.22), test: makeNoisyScatter(70, 8, 0.22) }),
    [],
  )
  const Xtr = useMemo(() => train.map((p) => [p.x]), [train])
  const ytr = useMemo(() => train.map((p) => p.y), [train])

  const { trainCurve, testCurve, bestB, yMax } = useMemo(() => {
    const model = trainGBRegressor(Xtr, ytr, { numTrees: ROUNDS, learningRate: lr, maxDepth: DEPTH, minLeaf: 3 })
    const trainCurve: number[] = []
    const testCurve: number[] = []
    for (let B = 1; B <= ROUNDS; B++) {
      let str = 0
      for (const p of train) str += (gbPredict(model, [p.x], B) - p.y) ** 2
      let ste = 0
      for (const p of test) ste += (gbPredict(model, [p.x], B) - p.y) ** 2
      trainCurve.push(str / train.length)
      testCurve.push(ste / test.length)
    }
    const bestB = testCurve.indexOf(Math.min(...testCurve)) + 1
    const yMax = Math.min(0.25, Math.max(...testCurve, ...trainCurve) * 1.05)
    return { trainCurve, testCurve, bestB, yMax }
  }, [Xtr, ytr, train, test, lr])

  const bx = (B: number) => PAD_L + ((B - 1) / (ROUNDS - 1)) * (W - PAD_L - PAD_R)
  const by = (e: number) => H - PAD_B - (Math.min(e, yMax) / yMax) * (H - PAD_T - PAD_B)
  const path = (arr: number[]) =>
    arr.map((e, i) => `${i === 0 ? 'M' : 'L'} ${bx(i + 1).toFixed(1)} ${by(e).toFixed(1)}`).join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Learning rate ν = <span className="font-mono">{lr.toFixed(2)}</span></span>
          <input type="range" min={0} max={1} step={0.01} value={t} onChange={(e) => setT(parseFloat(e.target.value))} className="w-44" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={PAD_L} y1={by(0)} x2={W - PAD_R} y2={by(0)} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={by(0)} stroke="var(--rule)" strokeWidth={1} />

          {/* early-stop marker */}
          <line x1={bx(bestB)} y1={PAD_T} x2={bx(bestB)} y2={by(0)} stroke="var(--accent)" strokeWidth={1} strokeDasharray="3,3" opacity={0.6} />
          <text x={bx(bestB)} y={PAD_T - 4} fontSize={10} fill="var(--accent)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">early stop</text>

          <path d={path(trainCurve)} fill="none" stroke="var(--ink-muted)" strokeWidth={1.75} strokeDasharray="4,3" />
          <path d={path(testCurve)} fill="none" stroke="var(--accent)" strokeWidth={2} />
          <circle cx={bx(bestB)} cy={by(testCurve[bestB - 1])} r={4} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1.25} />

          {/* y ticks */}
          {[0, yMax / 2, yMax].map((v, i) => (
            <text key={i} x={PAD_L - 6} y={by(v) + 3} fontSize={10} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{v.toFixed(2)}</text>
          ))}
          {[1, 40, 80, 120].map((B) => (
            <text key={B} x={bx(B)} y={by(0) + 16} fontSize={10} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{B}</text>
          ))}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 6} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">number of trees</text>
          <text x={PAD_L} y={PAD_T - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">error (MSE)</text>

          {/* legend */}
          <g transform={`translate(${W - PAD_R - 150}, ${PAD_T + 6})`}>
            <line x1={0} y1={0} x2={16} y2={0} stroke="var(--accent)" strokeWidth={2} />
            <text x={20} y={3.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">held-out error</text>
            <line x1={0} y1={14} x2={16} y2={14} stroke="var(--ink-muted)" strokeWidth={1.75} strokeDasharray="4,3" />
            <text x={20} y={17.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">training error</text>
          </g>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>early-stop round = <span className="text-accent">{bestB}</span></div>
        <div>best test MSE = <span className="text-ink">{testCurve[bestB - 1].toFixed(4)}</span></div>
        <div>test @ {ROUNDS} = <span className="text-ink">{testCurve[ROUNDS - 1].toFixed(4)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 14.3 — Boosting can overfit. Training error (grey, dashed)
        falls without limit as trees are added — the ensemble keeps fitting
        the training set better. But the held-out error (teal) bottoms out and
        turns back up: past the marked round, every new tree is fitting noise.
        Unlike a random forest, where more trees never hurt, a boosted model
        has an optimal number of trees, found by watching this curve and
        stopping at its minimum — early stopping, the discipline of Chapter
        11. Lower the learning rate and the minimum drops, but arrives later.
      </figcaption>
    </figure>
  )
}
