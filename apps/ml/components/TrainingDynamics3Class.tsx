'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  makeThreeBlobs,
  makeThreeOverlap,
  predictProb,
  trainMultinomial,
  type LabeledPoint,
  type MNFrame,
} from '@/lib/multinomial'

const DATASETS = {
  blobs: 'Blobs',
  overlap: 'Overlap',
} as const
type DatasetKey = keyof typeof DATASETS

function generate(key: DatasetKey, seed: number): LabeledPoint[] {
  switch (key) {
    case 'blobs':
      return makeThreeBlobs(120, seed)
    case 'overlap':
      return makeThreeOverlap(120, seed)
  }
}

const DW = 440
const DH = 360
const LW = 320
const LH = 360
const PAD = 26
const LOSS_PAD_L = 44
const LOSS_PAD_R = 18
const LOSS_PAD_TB = 26

const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.5
const Y_MAX = 2.5
const K = 3

const CLASS_RGB: [number, number, number][] = [
  [60, 90, 140],
  [199, 82, 42],
  [93, 138, 58],
]
const CLASS_HEX = ['#3c5a8c', '#c7522a', '#5d8a3a'] as const

function blendColor(probs: number[]): string {
  let r = 0
  let g = 0
  let b = 0
  for (let i = 0; i < probs.length; i++) {
    r += probs[i] * CLASS_RGB[i][0]
    g += probs[i] * CLASS_RGB[i][1]
    b += probs[i] * CLASS_RGB[i][2]
  }
  let max = -Infinity
  for (const p of probs) if (p > max) max = p
  const certainty = (max - 1 / K) / (1 - 1 / K)
  const alpha = Math.max(0, certainty) * 0.5
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

export default function TrainingDynamics3Class() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('blobs')
  const [seed, setSeed] = useState(2)
  const [lr, setLr] = useState(0.8)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const data = useMemo(() => generate(datasetKey, seed), [datasetKey, seed])

  const frames: MNFrame[] = useMemo(
    () =>
      trainMultinomial(data, {
        K: 3,
        lr,
        maxSteps: 120,
        tol: 1e-5,
      }),
    [data, lr],
  )

  useEffect(() => {
    setFrameIdx(0)
    setPlaying(false)
  }, [frames])

  useEffect(() => {
    if (!playing) return
    if (frameIdx >= frames.length - 1) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setFrameIdx((i) => i + 1), 110)
    return () => clearTimeout(t)
  }, [playing, frameIdx, frames.length])

  const cur = frames[Math.min(frameIdx, frames.length - 1)]
  const model = cur.model

  const sx = (x: number) =>
    PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (DW - 2 * PAD)
  const sy = (y: number) =>
    DH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (DH - 2 * PAD)

  const heatmap = useMemo(() => {
    const GX = 36
    const GY = 26
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cellW = (DW - 2 * PAD) / GX
    const cellH = (DH - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        // j = 0 is the top row of cells in SVG space; math y goes up,
        // so the top SVG row corresponds to the *largest* yD.
        const yD = Y_MAX - ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        const p = predictProb(model, [xD, yD])
        cells.push({
          x: PAD + i * cellW,
          y: PAD + j * cellH,
          w: cellW + 0.5,
          h: cellH + 0.5,
          fill: blendColor(p),
        })
      }
    }
    return cells
  }, [model])

  const boundaryLines = useMemo(() => {
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = []
    for (let c = 0; c < K; c++) {
      for (let d = c + 1; d < K; d++) {
        const a = model.W[c][0] - model.W[d][0]
        const b1 = model.W[c][1] - model.W[d][1]
        const c1 = model.b[c] - model.b[d]
        const len2 = a * a + b1 * b1
        if (len2 < 1e-12) continue
        const cx = (-c1 * a) / len2
        const cy = (-c1 * b1) / len2
        const len = Math.sqrt(len2)
        const dxL = -b1 / len
        const dyL = a / len
        const t = 100
        lines.push({
          x1: cx - t * dxL,
          y1: cy - t * dyL,
          x2: cx + t * dxL,
          y2: cy + t * dyL,
        })
      }
    }
    return lines
  }, [model])

  const lossMax = useMemo(
    () => Math.max(...frames.map((f) => f.loss)) * 1.05,
    [frames],
  )
  const lossMin = 0

  const lx = (step: number) =>
    LOSS_PAD_L +
    (step / Math.max(frames.length - 1, 1)) * (LW - LOSS_PAD_L - LOSS_PAD_R)
  const ly = (loss: number) =>
    LH -
    LOSS_PAD_TB -
    ((loss - lossMin) / (lossMax - lossMin)) * (LH - 2 * LOSS_PAD_TB)

  const lossPath = useMemo(() => {
    return frames
      .slice(0, frameIdx + 1)
      .map(
        (f, i) =>
          `${i === 0 ? 'M' : 'L'} ${lx(f.step).toFixed(2)} ${ly(f.loss).toFixed(2)}`,
      )
      .join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames, frameIdx, lossMax])

  const log3 = Math.log(3)

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Data</span>
          <select
            value={datasetKey}
            onChange={(e) => setDatasetKey(e.target.value as DatasetKey)}
            className="bg-paper border border-rule rounded px-2 py-1 text-ink"
          >
            {Object.entries(DATASETS).map(([k, label]) => (
              <option key={k} value={k}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          New seed
        </button>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Learning rate ={' '}
            <span className="font-mono">{lr.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.05}
            value={lr}
            onChange={(e) => setLr(parseFloat(e.target.value))}
            className="w-32"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <div className="flex flex-col md:flex-row">
          {/* DATA PANEL */}
          <svg
            viewBox={`0 0 ${DW} ${DH}`}
            className="w-full md:w-[58%] h-auto block border-b md:border-b-0 md:border-r border-rule"
          >
            {heatmap.map((c, i) => (
              <rect
                key={i}
                x={c.x}
                y={c.y}
                width={c.w}
                height={c.h}
                fill={c.fill}
              />
            ))}

            {boundaryLines.map((ln, i) => (
              <line
                key={i}
                x1={sx(ln.x1)}
                y1={sy(ln.y1)}
                x2={sx(ln.x2)}
                y2={sy(ln.y2)}
                stroke="var(--ink)"
                strokeWidth={1.2}
                opacity={0.55}
                strokeDasharray="4,3"
              />
            ))}

            {data.map((s, i) => (
              <circle
                key={i}
                cx={sx(s.x[0])}
                cy={sy(s.x[1])}
                r={3.5}
                fill={CLASS_HEX[s.y]}
                stroke="var(--paper)"
                strokeWidth={0.8}
              />
            ))}

            <text
              x={PAD}
              y={PAD - 6}
              fontSize={11}
              fill="var(--ink-muted)"
              fontFamily="var(--font-sans, sans-serif)"
            >
              data space
            </text>
          </svg>

          {/* LOSS PANEL */}
          <svg
            viewBox={`0 0 ${LW} ${LH}`}
            className="w-full md:w-[42%] h-auto block"
          >
            <line
              x1={LOSS_PAD_L}
              y1={ly(lossMin)}
              x2={LW - LOSS_PAD_R}
              y2={ly(lossMin)}
              stroke="var(--rule)"
              strokeWidth={1}
            />
            <line
              x1={LOSS_PAD_L}
              y1={ly(lossMin)}
              x2={LOSS_PAD_L}
              y2={ly(lossMax)}
              stroke="var(--rule)"
              strokeWidth={1}
            />
            <text
              x={LOSS_PAD_L - 6}
              y={ly(lossMax) + 4}
              fontSize={10}
              fill="var(--ink-muted)"
              textAnchor="end"
              fontFamily="var(--font-mono, monospace)"
            >
              {lossMax.toFixed(2)}
            </text>
            <text
              x={LOSS_PAD_L - 6}
              y={ly(lossMin) + 4}
              fontSize={10}
              fill="var(--ink-muted)"
              textAnchor="end"
              fontFamily="var(--font-mono, monospace)"
            >
              0
            </text>
            <text
              x={LW - LOSS_PAD_R}
              y={ly(lossMin) + 14}
              fontSize={10}
              fill="var(--ink-muted)"
              textAnchor="end"
            >
              step
            </text>
            <text
              x={LOSS_PAD_L}
              y={PAD - 6}
              fontSize={11}
              fill="var(--ink-muted)"
              fontFamily="var(--font-sans, sans-serif)"
            >
              categorical cross-entropy
            </text>

            {/* log 3 baseline reference */}
            {log3 < lossMax && (
              <>
                <line
                  x1={LOSS_PAD_L}
                  y1={ly(log3)}
                  x2={LW - LOSS_PAD_R}
                  y2={ly(log3)}
                  stroke="var(--ink-muted)"
                  strokeWidth={1}
                  strokeDasharray="2,3"
                  opacity={0.45}
                />
                <text
                  x={LW - LOSS_PAD_R - 4}
                  y={ly(log3) - 4}
                  fontSize={9}
                  fill="var(--ink-muted)"
                  textAnchor="end"
                  fontFamily="var(--font-mono, monospace)"
                  opacity={0.7}
                >
                  log 3
                </text>
              </>
            )}

            <path
              d={frames
                .map(
                  (f, i) =>
                    `${i === 0 ? 'M' : 'L'} ${lx(f.step).toFixed(2)} ${ly(f.loss).toFixed(2)}`,
                )
                .join(' ')}
              fill="none"
              stroke="var(--ink-muted)"
              strokeWidth={1}
              opacity={0.2}
            />
            <path
              d={lossPath}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.75}
            />
            <circle
              cx={lx(cur.step)}
              cy={ly(cur.loss)}
              r={4}
              fill="var(--accent)"
              stroke="var(--paper)"
              strokeWidth={1.5}
            />
          </svg>
        </div>
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button
          onClick={() => {
            if (frameIdx >= frames.length - 1) setFrameIdx(0)
            setPlaying((p) => !p)
          }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[60px]"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => {
            setFrameIdx(0)
            setPlaying(false)
          }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Reset
        </button>
        <button
          onClick={() =>
            setFrameIdx((i) => Math.min(i + 1, frames.length - 1))
          }
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Step
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 1)}
          value={frameIdx}
          onChange={(e) => {
            setPlaying(false)
            setFrameIdx(parseInt(e.target.value))
          }}
          className="flex-1"
        />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">
          step {frameIdx} / {frames.length - 1}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>loss = {cur.loss.toFixed(4)}</div>
        <div>log 3 ≈ {log3.toFixed(4)} (random)</div>
        <div>
          accuracy ={' '}
          <span className="text-ink">{(cur.accuracy * 100).toFixed(1)}%</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 8.3 — Training multinomial logistic regression by gradient
        descent. On the left, the three decision regions form as the
        pairwise boundaries slide into place; on the right, categorical
        cross-entropy falls from log 3 (the random-guess baseline) toward
        zero. Each step on the right is one gradient update, producing one
        new partition on the left.
      </figcaption>
    </figure>
  )
}
