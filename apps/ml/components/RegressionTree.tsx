'use client'

import { useMemo, useState } from 'react'
import { makeNoisyScatter, trueCurve } from '@/lib/regularisation'
import {
  buildRegressionTree,
  predictReg,
  regSteps,
  countLeavesReg,
} from '@/lib/tree'

const W = 600
const H = 320
const PAD = 30
const X_MIN = -1
const X_MAX = 1
const Y_MIN = -1.5
const Y_MAX = 1.5

export default function RegressionTree() {
  const [depth, setDepth] = useState(3)

  const { train, test } = useMemo(
    () => ({
      train: makeNoisyScatter(60, 3, 0.18).map((p) => ({ x: p.x, y: p.y })),
      test: makeNoisyScatter(60, 8, 0.18).map((p) => ({ x: p.x, y: p.y })),
    }),
    [],
  )

  const tree = useMemo(
    () => buildRegressionTree(train.map((p) => ({ x: [p.x], y: p.y })), depth, 2),
    [train, depth],
  )
  const steps = useMemo(() => regSteps(tree, X_MIN, X_MAX), [tree])
  const leaves = countLeavesReg(tree)

  const trainMSE = useMemo(() => {
    let s = 0
    for (const p of train) s += (predictReg(tree, [p.x]) - p.y) ** 2
    return s / train.length
  }, [tree, train])
  const testMSE = useMemo(() => {
    let s = 0
    for (const p of test) s += (predictReg(tree, [p.x]) - p.y) ** 2
    return s / test.length
  }, [tree, test])

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (y: number) =>
    H - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD)
  const clampY = (y: number) => Math.max(Y_MIN, Math.min(Y_MAX, y))

  // staircase path: horizontal segment per leaf, vertical jumps between
  const stepPath = useMemo(() => {
    const parts: string[] = []
    steps.forEach((s, i) => {
      const yv = clampY(s.value)
      if (i === 0) parts.push(`M ${sx(s.x0).toFixed(1)} ${sy(yv).toFixed(1)}`)
      else parts.push(`L ${sx(s.x0).toFixed(1)} ${sy(yv).toFixed(1)}`)
      parts.push(`L ${sx(s.x1).toFixed(1)} ${sy(yv).toFixed(1)}`)
    })
    return parts.join(' ')
  }, [steps])

  const truePath = useMemo(() => {
    const N = 200
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(trueCurve(x)).toFixed(1)}`)
    }
    return parts.join(' ')
  }, [])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Max depth = <span className="font-mono">{depth}</span>
          </span>
          <input
            type="range"
            min={0}
            max={8}
            step={1}
            value={depth}
            onChange={(e) => setDepth(parseInt(e.target.value))}
            className="w-48"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* zero axis */}
          <line x1={sx(X_MIN)} y1={sy(0)} x2={sx(X_MAX)} y2={sy(0)} stroke="var(--rule)" strokeWidth={1} strokeDasharray="2,3" />

          {/* true curve */}
          <path d={truePath} fill="none" stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7} />

          {/* data points */}
          {train.map((p, i) => (
            <circle key={i} cx={sx(p.x)} cy={sy(clampY(p.y))} r={3} fill="var(--ink)" stroke="var(--paper)" strokeWidth={0.8} opacity={0.55} />
          ))}

          {/* step function */}
          <path d={stepPath} fill="none" stroke="var(--accent)" strokeWidth={2.25} />

          {/* labels */}
          <text x={W - PAD} y={sy(0) - 6} fontSize={11} fill="var(--ink-muted)" textAnchor="end" fontStyle="italic">x</text>
          <text x={sx(0) + 8} y={sy(Y_MAX) + 12} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">y</text>
          <text x={PAD + 2} y={PAD + 4} fontSize={11} fill="var(--ink-muted)" fontStyle="italic" fontFamily="var(--font-sans, sans-serif)">
            regression tree = piecewise-constant fit ({leaves} {leaves === 1 ? 'leaf' : 'leaves'})
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>leaves (pieces) = <span className="text-ink">{leaves}</span></div>
        <div>train MSE = <span className="text-ink">{trainMSE.toFixed(4)}</span></div>
        <div>test MSE = <span style={{ color: 'var(--accent)' }}>{testMSE.toFixed(4)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 12.4 — A regression tree fitting the smooth curve from Chapter 9
        (dashed). It splits on x to reduce variance and predicts the mean of
        each leaf, so its output is a staircase. At depth 0 it is a single
        flat line at the overall mean; raise the depth and the steps multiply
        and narrow. Push it too far and the test error climbs back up — the
        staircase has started fitting the noise, the regression face of the
        same overfitting you saw with classification depth.
      </figcaption>
    </figure>
  )
}
