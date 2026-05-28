'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  makeBlobs,
  makeMoons,
  makeOverlap,
  sigmoid,
  trainLogistic,
  type LabeledPoint,
  type LRFrame,
} from '@/lib/logistic'

const DATASETS = {
  blobs: 'Blobs',
  overlap: 'Overlap',
  moons: 'Moons',
} as const
type DatasetKey = keyof typeof DATASETS

function generate(key: DatasetKey, seed: number): LabeledPoint[] {
  switch (key) {
    case 'blobs':
      return makeBlobs(120, seed)
    case 'overlap':
      return makeOverlap(120, seed)
    case 'moons':
      return makeMoons(120, seed)
  }
}

const DW = 440 // data panel width
const DH = 360
const LW = 320 // loss panel width
const LH = 360
const PAD = 26
const LOSS_PAD_L = 44
const LOSS_PAD_R = 18
const LOSS_PAD_TB = 26

const X_MIN = -3
const X_MAX = 3
const Y_MIN = -2.2
const Y_MAX = 2.2

export default function TrainingDynamics() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('blobs')
  const [seed, setSeed] = useState(2)
  const [lr, setLr] = useState(0.5)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const data = useMemo(() => generate(datasetKey, seed), [datasetKey, seed])

  // Train from a deterministic but non-trivial initialisation so the
  // trajectory has visible motion. Different seeds give different starts.
  const frames: LRFrame[] = useMemo(() => {
    // Initialise with a rotated, off-centre boundary so the descent has
    // a story to tell. Seed-dependent so "New seed" reshuffles both data
    // and starting point.
    const a = (seed * 1.3) % (Math.PI * 2)
    const w0: [number, number] = [Math.cos(a) * 0.4, Math.sin(a) * 0.4]
    const b0 = (((seed * 7) % 11) - 5) * 0.2
    return trainLogistic(data, { lr, maxSteps: 80, w0, b0, tol: 1e-5 })
  }, [data, lr, seed])

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
    const t = setTimeout(() => setFrameIdx((i) => i + 1), 130)
    return () => clearTimeout(t)
  }, [playing, frameIdx, frames.length])

  const cur = frames[Math.min(frameIdx, frames.length - 1)]
  const model = cur.model

  // ─── Data-space transforms ─────────────────────────────────────────
  const sx = (x: number) =>
    PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (DW - 2 * PAD)
  const sy = (y: number) =>
    DH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (DH - 2 * PAD)

  // Heatmap (recomputed each frame)
  const heatmap = useMemo(() => {
    const GX = 40
    const GY = 28
    const cells: Array<{ x: number; y: number; w: number; h: number; fill: string }> = []
    const cellW = (DW - 2 * PAD) / GX
    const cellH = (DH - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yD = Y_MIN + ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        const z = model.w[0] * xD + model.w[1] * yD + model.b
        const p = sigmoid(z)
        let fill: string
        if (p < 0.5) {
          const a = (0.5 - p) * 2 * 0.4
          fill = `rgba(60, 90, 140, ${a.toFixed(3)})`
        } else {
          const a = (p - 0.5) * 2 * 0.4
          fill = `rgba(199, 82, 42, ${a.toFixed(3)})`
        }
        cells.push({
          x: PAD + i * cellW,
          y: PAD + j * cellH,
          w: cellW + 0.5,
          h: cellH + 0.5,
          fill,
        })
      }
    }
    return cells
  }, [model])

  // Boundary line endpoints clipped far outside the viewport
  const lineEnds = useMemo(() => {
    const wnorm = Math.hypot(model.w[0], model.w[1]) || 1e-6
    // Closest point on line to origin
    const cx = (-model.b * model.w[0]) / (wnorm * wnorm)
    const cy = (-model.b * model.w[1]) / (wnorm * wnorm)
    // Along-line direction (perpendicular to w)
    const dx = -model.w[1] / wnorm
    const dy = model.w[0] / wnorm
    const t = 100
    return [
      [cx - t * dx, cy - t * dy] as [number, number],
      [cx + t * dx, cy + t * dy] as [number, number],
    ]
  }, [model])

  // ─── Loss-curve transforms ─────────────────────────────────────────
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
            Learning rate = <span className="font-mono">{lr.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.05}
            max={2}
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
            {/* heatmap */}
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

            {/* boundary */}
            <line
              x1={sx(lineEnds[0][0])}
              y1={sy(lineEnds[0][1])}
              x2={sx(lineEnds[1][0])}
              y2={sy(lineEnds[1][1])}
              stroke="var(--ink)"
              strokeWidth={1.5}
              opacity={0.6}
              strokeDasharray="4,3"
            />

            {/* points */}
            {data.map((s, i) => (
              <circle
                key={i}
                cx={sx(s.x[0])}
                cy={sy(s.x[1])}
                r={3.5}
                fill={s.y === 1 ? '#c7522a' : '#3c5a8c'}
                stroke="var(--paper)"
                strokeWidth={0.8}
              />
            ))}

            {/* panel label */}
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
            {/* axes */}
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

            {/* tick labels */}
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
              cross-entropy loss
            </text>

            {/* full loss curve, ghosted */}
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

            {/* progressed loss curve */}
            <path
              d={lossPath}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.75}
            />

            {/* current step marker */}
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

      {/* Playback */}
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
          onClick={() => setFrameIdx((i) => Math.min(i + 1, frames.length - 1))}
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

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          w = ({model.w[0].toFixed(2)}, {model.w[1].toFixed(2)})
        </div>
        <div>b = {model.b.toFixed(2)}</div>
        <div>loss = {cur.loss.toFixed(4)}</div>
        <div>
          accuracy ={' '}
          <span className="text-ink">{(cur.accuracy * 100).toFixed(1)}%</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 7.3 — Training logistic regression with gradient descent. On
        the left, the decision boundary rotates and translates in data
        space; on the right, the cross-entropy loss falls. Both panels move
        in lockstep — each step on the right is one gradient update,
        producing one new boundary on the left.
      </figcaption>
    </figure>
  )
}
