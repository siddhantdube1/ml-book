'use client'

import { useMemo, useState } from 'react'
import { makeNoisyScatter, trueCurve } from '@/lib/regularisation'
import { trainGBRegressor, gbPredict } from '@/lib/boosting'

const W = 600
const H = 320
const PAD = 30
const X_MIN = -1
const X_MAX = 1
const Y_MIN = -1.5
const Y_MAX = 1.5
const ROUNDS = 60
const DEPTH = 2
const LR_MIN = 0.02
const LR_MAX = 1

export default function LearningRate() {
  const [t, setT] = useState(0.45) // slider position -> log lr

  const lr = useMemo(() => {
    const lo = Math.log10(LR_MIN)
    const hi = Math.log10(LR_MAX)
    return Math.pow(10, lo + t * (hi - lo))
  }, [t])

  const { train, test } = useMemo(
    () => ({ train: makeNoisyScatter(80, 3, 0.18), test: makeNoisyScatter(80, 8, 0.18) }),
    [],
  )
  const Xtr = useMemo(() => train.map((p) => [p.x]), [train])
  const ytr = useMemo(() => train.map((p) => p.y), [train])

  const model = useMemo(
    () => trainGBRegressor(Xtr, ytr, { numTrees: ROUNDS, learningRate: lr, maxDepth: DEPTH, minLeaf: 3 }),
    [Xtr, ytr, lr],
  )

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (v: number) => H - PAD - ((v - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD)
  const clampY = (v: number) => Math.max(Y_MIN, Math.min(Y_MAX, v))

  const fitPath = useMemo(() => {
    const N = 280
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(clampY(gbPredict(model, [x]))).toFixed(1)}`)
    }
    return parts.join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model])

  const truePath = useMemo(() => {
    const N = 200
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(trueCurve(x)).toFixed(1)}`)
    }
    return parts.join(' ')
  }, [])

  const trainMSE = useMemo(() => {
    let s = 0
    for (const p of train) s += (gbPredict(model, [p.x]) - p.y) ** 2
    return s / train.length
  }, [train, model])
  const testMSE = useMemo(() => {
    let s = 0
    for (const p of test) s += (gbPredict(model, [p.x]) - p.y) ** 2
    return s / test.length
  }, [test, model])

  const label = lr < 0.05 ? 'underfit — steps too timid' : lr > 0.5 ? 'overfit — steps too large' : 'good fit'

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Learning rate ν = <span className="font-mono">{lr.toFixed(2)}</span>
          </span>
          <input type="range" min={0} max={1} step={0.01} value={t} onChange={(e) => setT(parseFloat(e.target.value))} className="w-48" />
        </label>
        <span className="text-ink-muted">({ROUNDS} trees, depth {DEPTH})</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <line x1={sx(X_MIN)} y1={sy(0)} x2={sx(X_MAX)} y2={sy(0)} stroke="var(--rule)" strokeWidth={1} strokeDasharray="2,3" />
          <path d={truePath} fill="none" stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.6} />
          {train.map((p, i) => (
            <circle key={i} cx={sx(p.x)} cy={sy(clampY(p.y))} r={2.6} fill="var(--ink)" stroke="var(--paper)" strokeWidth={0.6} opacity={0.45} />
          ))}
          <path d={fitPath} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
          <text x={PAD} y={PAD - 6} fontSize={11} fill="var(--ink-muted)" fontStyle="italic" fontFamily="var(--font-sans, sans-serif)">
            boosted fit at ν = {lr.toFixed(2)} — {label}
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>ν = {lr.toFixed(3)}</div>
        <div>train MSE = <span className="text-ink">{trainMSE.toFixed(4)}</span></div>
        <div>test MSE = <span style={{ color: 'var(--accent)' }}>{testMSE.toFixed(4)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 14.2 — Shrinkage. The same sixty trees, but each added at
        strength ν. Turn ν down and the sixty timid steps barely leave the
        mean — the model underfits. Turn it up toward one and each tree slams
        in at full force, and the fit turns jagged, chasing the noise. Around
        ν = 0.1 the many small steps track the true curve cleanly: in
        boosting, patience generalises.
      </figcaption>
    </figure>
  )
}
