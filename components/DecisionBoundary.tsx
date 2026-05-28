'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  makeBlobs,
  makeMoons,
  makeOverlap,
  sigmoid,
  accuracy,
  type LabeledPoint,
  type LRModel,
} from '@/lib/logistic'

const W = 640
const H = 420
const PAD = 24

const DATASETS = {
  blobs: 'Blobs — linearly separable',
  overlap: 'Overlap — partially separable',
  moons: 'Moons — not linearly separable',
} as const
type DatasetKey = keyof typeof DATASETS

function generate(key: DatasetKey, seed: number): LabeledPoint[] {
  switch (key) {
    case 'blobs':
      return makeBlobs(80, seed)
    case 'overlap':
      return makeOverlap(80, seed)
    case 'moons':
      return makeMoons(80, seed)
  }
}

export default function DecisionBoundary() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('blobs')
  const [seed, setSeed] = useState(1)
  const [sharpness, setSharpness] = useState(1.5)
  // Two handle positions in data coordinates
  const [handle1, setHandle1] = useState<[number, number]>([0, 1.5])
  const [handle2, setHandle2] = useState<[number, number]>([0, -1.5])
  const [dragging, setDragging] = useState<0 | 1 | null>(null)

  const data = useMemo(() => generate(datasetKey, seed), [datasetKey, seed])

  // Data domain — fixed for clean axes; data is always centred near origin
  const X_MIN = -3
  const X_MAX = 3
  const Y_MIN = -2.2
  const Y_MAX = 2.2

  const sx = (x: number) =>
    PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (y: number) =>
    H - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD)
  const invX = (px: number) =>
    X_MIN + ((px - PAD) / (W - 2 * PAD)) * (X_MAX - X_MIN)
  const invY = (py: number) =>
    Y_MIN + ((H - PAD - py) / (H - 2 * PAD)) * (Y_MAX - Y_MIN)

  // Derive the LR model from the two handle positions and the sharpness.
  // Line direction = handle2 - handle1. Normal n = 90° CCW rotation of that,
  // normalized. Then w = sharpness · n (scaled normal), and b chosen so the
  // line w · x + b = 0 passes through handle1.
  const model: LRModel = useMemo(() => {
    const dx = handle2[0] - handle1[0]
    const dy = handle2[1] - handle1[1]
    const len = Math.hypot(dx, dy) || 1
    const nx = -dy / len
    const ny = dx / len
    const w: [number, number] = [sharpness * nx, sharpness * ny]
    const b = -(w[0] * handle1[0] + w[1] * handle1[1])
    return { w, b }
  }, [handle1, handle2, sharpness])

  // Probability heatmap: sample on a grid, render coloured rectangles
  const heatmap = useMemo(() => {
    const GX = 48
    const GY = 32
    const cells: Array<{ x: number; y: number; w: number; h: number; fill: string }> = []
    const cellW = (W - 2 * PAD) / GX
    const cellH = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xData = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yData = Y_MIN + ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        const z = model.w[0] * xData + model.w[1] * yData + model.b
        const p = sigmoid(z)
        // Map probability to a colour: blue (class 0) ↔ paper ↔ orange (class 1).
        // Use rgba so the paper colour shows through naturally near p = 0.5.
        let fill: string
        if (p < 0.5) {
          const a = (0.5 - p) * 2 * 0.42
          fill = `rgba(60, 90, 140, ${a.toFixed(3)})`
        } else {
          const a = (p - 0.5) * 2 * 0.42
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

  // Accuracy at threshold 0.5
  const acc = useMemo(() => accuracy(data, model), [data, model])

  // Pointer handlers for the boundary handles
  const svgRef = useRef<SVGSVGElement>(null)
  function svgPoint(e: React.PointerEvent<SVGSVGElement>): [number, number] | null {
    const svg = svgRef.current
    if (!svg) return null
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    return [x, y]
  }
  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (dragging === null) return
    const p = svgPoint(e)
    if (!p) return
    const dx = invX(p[0])
    const dy = invY(p[1])
    const clamped: [number, number] = [
      Math.max(X_MIN, Math.min(X_MAX, dx)),
      Math.max(Y_MIN, Math.min(Y_MAX, dy)),
    ]
    if (dragging === 0) setHandle1(clamped)
    else setHandle2(clamped)
  }

  // Compute boundary line endpoints clipped to the viewport
  const lineEndpoints = useMemo(() => {
    const dx = handle2[0] - handle1[0]
    const dy = handle2[1] - handle1[1]
    const len = Math.hypot(dx, dy) || 1
    const ux = dx / len
    const uy = dy / len
    // Extend far beyond the viewport in both directions
    const t = 100
    return [
      [handle1[0] - t * ux, handle1[1] - t * uy] as [number, number],
      [handle2[0] + t * ux, handle2[1] + t * uy] as [number, number],
    ]
  }, [handle1, handle2])

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
            Sharpness ‖w‖ = <span className="font-mono">{sharpness.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.3}
            max={6}
            step={0.05}
            value={sharpness}
            onChange={(e) => setSharpness(parseFloat(e.target.value))}
            className="w-40"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block select-none"
          onPointerMove={onPointerMove}
          onPointerUp={(e) => {
            setDragging(null)
            e.currentTarget.releasePointerCapture(e.pointerId)
          }}
        >
          {/* probability heatmap */}
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

          {/* boundary line */}
          <line
            x1={sx(lineEndpoints[0][0])}
            y1={sy(lineEndpoints[0][1])}
            x2={sx(lineEndpoints[1][0])}
            y2={sy(lineEndpoints[1][1])}
            stroke="var(--ink)"
            strokeWidth={1.5}
            opacity={0.55}
            strokeDasharray="4,3"
          />

          {/* data points */}
          {data.map((s, i) => (
            <circle
              key={i}
              cx={sx(s.x[0])}
              cy={sy(s.x[1])}
              r={4}
              fill={s.y === 1 ? '#c7522a' : '#3c5a8c'}
              stroke="var(--paper)"
              strokeWidth={1}
            />
          ))}

          {/* boundary handles */}
          {[handle1, handle2].map((h, i) => (
            <circle
              key={i}
              cx={sx(h[0])}
              cy={sy(h[1])}
              r={8}
              fill="var(--paper)"
              stroke="var(--ink)"
              strokeWidth={2}
              className="cursor-grab"
              onPointerDown={(e) => {
                setDragging(i as 0 | 1)
                e.currentTarget.setPointerCapture(e.pointerId)
                e.stopPropagation()
              }}
            />
          ))}
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          w = ({model.w[0].toFixed(2)}, {model.w[1].toFixed(2)})
        </div>
        <div>b = {model.b.toFixed(2)}</div>
        <div>
          accuracy ={' '}
          <span className="text-ink">{(acc * 100).toFixed(1)}%</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 7.2 — A logistic regression model on 2D data. Drag the two
        handles to move the decision boundary; the heatmap shows the model's
        probability that each point belongs to class 1 (orange). Sharper{' '}
        <span className="font-mono">‖w‖</span> compresses the transition zone
        around the boundary.
      </figcaption>
    </figure>
  )
}
