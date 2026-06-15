'use client'

import { useEffect, useMemo, useState } from 'react'
import { makeNoisyScatter, trueCurve } from '@/lib/regularisation'
import { predictReg } from '@/lib/tree'
import { trainGBRegressor, gbPredict } from '@/lib/boosting'

const W = 600
const TOP_H = 230
const BOT_H = 150
const GAP = 10
const H = TOP_H + BOT_H + GAP
const PAD_X = 32
const PAD_Y = 22
const X_MIN = -1
const X_MAX = 1
const Y_MIN = -1.5
const Y_MAX = 1.5
const R_MAX = 0.7 // residual axis half-range
const ROUNDS = 40
const LR = 0.3
const DEPTH = 2

export default function ResidualBoosting() {
  const [round, setRound] = useState(0)
  const [playing, setPlaying] = useState(false)

  const data = useMemo(() => makeNoisyScatter(80, 3, 0.18), [])
  const X = useMemo(() => data.map((p) => [p.x]), [data])
  const y = useMemo(() => data.map((p) => p.y), [data])

  const model = useMemo(
    () => trainGBRegressor(X, y, { numTrees: ROUNDS, learningRate: LR, maxDepth: DEPTH, minLeaf: 3 }),
    [X, y],
  )

  useEffect(() => {
    if (!playing) return
    if (round >= ROUNDS) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setRound((r) => r + 1), 320)
    return () => clearTimeout(t)
  }, [playing, round])

  const sx = (x: number) => PAD_X + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD_X)
  const syTop = (v: number) => PAD_Y + ((Y_MAX - v) / (Y_MAX - Y_MIN)) * (TOP_H - 2 * PAD_Y)
  const botBase = TOP_H + GAP
  const syBot = (v: number) =>
    botBase + PAD_Y + ((R_MAX - v) / (2 * R_MAX)) * (BOT_H - 2 * PAD_Y)
  const clampT = (v: number) => Math.max(Y_MIN, Math.min(Y_MAX, v))
  const clampR = (v: number) => Math.max(-R_MAX, Math.min(R_MAX, v))

  // ensemble prediction curve at this round
  const predPath = useMemo(() => {
    const N = 240
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      const v = clampT(gbPredict(model, [x], round))
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${syTop(v).toFixed(1)}`)
    }
    return parts.join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, round])

  const truePath = useMemo(() => {
    const N = 200
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${syTop(trueCurve(x)).toFixed(1)}`)
    }
    return parts.join(' ')
  }, [])

  // residuals at this round + the next tree's fit through them
  const residuals = useMemo(
    () => data.map((p) => ({ x: p.x, r: p.y - gbPredict(model, [p.x], round) })),
    [data, model, round],
  )
  const nextTreePath = useMemo(() => {
    if (round >= ROUNDS) return ''
    const tree = model.trees[round]
    const N = 240
    const parts: string[] = []
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + (i / N) * (X_MAX - X_MIN)
      const v = clampR(predictReg(tree, [x]))
      parts.push(`${i === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${syBot(v).toFixed(1)}`)
    }
    return parts.join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, round])

  const trainMSE = useMemo(() => {
    let s = 0
    for (const p of data) s += (gbPredict(model, [p.x], round) - p.y) ** 2
    return s / data.length
  }, [data, model, round])

  return (
    <figure className="my-10">
      <div className="flex items-center gap-3 mb-4 font-sans text-sm">
        <button
          onClick={() => { if (round >= ROUNDS) setRound(0); setPlaying((p) => !p) }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[60px]"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={() => { setRound(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <button onClick={() => setRound((r) => Math.min(r + 1, ROUNDS))} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Add tree</button>
        <input type="range" min={0} max={ROUNDS} value={round} onChange={(e) => { setPlaying(false); setRound(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">trees: {round}</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* TOP: data + true curve + ensemble prediction */}
          <line x1={sx(X_MIN)} y1={syTop(0)} x2={sx(X_MAX)} y2={syTop(0)} stroke="var(--rule)" strokeWidth={1} strokeDasharray="2,3" />
          <path d={truePath} fill="none" stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.6} />
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x)} cy={syTop(clampT(p.y))} r={2.5} fill="var(--ink)" stroke="var(--paper)" strokeWidth={0.6} opacity={0.45} />
          ))}
          <path d={predPath} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
          <text x={PAD_X} y={PAD_Y - 6} fontSize={11} fill="var(--ink-muted)" fontStyle="italic" fontFamily="var(--font-sans, sans-serif)">
            data &amp; ensemble prediction F (sum of {round} trees)
          </text>

          {/* divider */}
          <line x1={0} y1={botBase} x2={W} y2={botBase} stroke="var(--rule)" strokeWidth={1} />

          {/* BOTTOM: residuals + next tree */}
          <line x1={sx(X_MIN)} y1={syBot(0)} x2={sx(X_MAX)} y2={syBot(0)} stroke="var(--rule)" strokeWidth={1} />
          {residuals.map((p, i) => (
            <circle key={i} cx={sx(p.x)} cy={syBot(clampR(p.r))} r={2.5} fill="#c7522a" stroke="var(--paper)" strokeWidth={0.5} opacity={0.7} />
          ))}
          {nextTreePath && <path d={nextTreePath} fill="none" stroke="#c7522a" strokeWidth={2} />}
          <text x={PAD_X} y={botBase + PAD_Y - 6} fontSize={11} fill="var(--ink-muted)" fontStyle="italic" fontFamily="var(--font-sans, sans-serif)">
            residuals (still-wrong amount) &amp; the next tree fit to them
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>trees so far = <span className="text-ink">{round}</span></div>
        <div>training MSE = <span className="text-accent">{trainMSE.toFixed(4)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 14.1 — Boosting, one tree at a time. The top panel is the data
        (dots), the true curve (dashed), and the ensemble's prediction (teal)
        — which starts as a flat line at the mean and climbs toward the data.
        The bottom panel is the residuals, the part each point still gets
        wrong, with the next little tree fit straight through them. Add a tree
        and that tree's correction is folded into the prediction above; the
        residuals below shrink toward zero. The sum of many small corrections
        converges on a curve no single shallow tree could fit.
      </figcaption>
    </figure>
  )
}
