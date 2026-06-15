'use client'

import { useMemo, useRef, useState } from 'react'
import {
  makeThreeBlobs,
  makeThreeOverlap,
  predictProb,
  predict,
  type LabeledPoint,
  type MNModel,
} from '@/lib/multinomial'

const W = 640
const H = 420
const PAD = 24
const K = 3

const CLASS_RGB: [number, number, number][] = [
  [60, 90, 140], // class 0 blue
  [199, 82, 42], // class 1 orange
  [93, 138, 58], // class 2 olive
]
const CLASS_HEX = ['#3c5a8c', '#c7522a', '#5d8a3a'] as const

const DATASETS = {
  blobs: 'Blobs — well separated',
  overlap: 'Overlap — wider classes',
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

const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.5
const Y_MAX = 2.5

/**
 * Blend the three class colours weighted by softmax probabilities. The
 * alpha rises with the *certainty* of the prediction — uniform predictions
 * (1/K each) are transparent, peaked predictions are saturated.
 */
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
  const alpha = Math.max(0, certainty) * 0.55
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

export default function MultiClassBoundary() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('blobs')
  const [seed, setSeed] = useState(1)
  const [sharpness, setSharpness] = useState(1.5)
  const [prototypes, setPrototypes] = useState<[number, number][]>([
    [-1.6, -0.8],
    [1.6, -0.8],
    [0, 1.4],
  ])
  const [dragging, setDragging] = useState<number | null>(null)

  const data = useMemo(() => generate(datasetKey, seed), [datasetKey, seed])

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (y: number) =>
    H - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - 2 * PAD)
  const invX = (px: number) =>
    X_MIN + ((px - PAD) / (W - 2 * PAD)) * (X_MAX - X_MIN)
  const invY = (py: number) =>
    Y_MIN + ((H - PAD - py) / (H - 2 * PAD)) * (Y_MAX - Y_MIN)

  // Model: W_c = sharpness * prototype_c, b_c = -½ sharpness ‖prototype_c‖².
  // This is the Gaussian-LDA construction; pairwise boundaries are the
  // perpendicular bisectors of the prototypes — a Voronoi partition.
  const model: MNModel = useMemo(() => {
    const Wmat: number[][] = prototypes.map((p) => [
      sharpness * p[0],
      sharpness * p[1],
    ])
    const bvec: number[] = prototypes.map(
      (p) => -0.5 * sharpness * (p[0] * p[0] + p[1] * p[1]),
    )
    return { W: Wmat, b: bvec }
  }, [prototypes, sharpness])

  const heatmap = useMemo(() => {
    const GX = 48
    const GY = 32
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cellW = (W - 2 * PAD) / GX
    const cellH = (H - 2 * PAD) / GY
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

  // Three pairwise boundary lines, drawn as dashed segments. Each is the
  // locus where two specific classes have equal score.
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

  const acc = useMemo(() => {
    let correct = 0
    for (const s of data) if (predict(model, s.x) === s.y) correct++
    return correct / data.length
  }, [data, model])

  const svgRef = useRef<SVGSVGElement>(null)
  function svgPoint(
    e: React.PointerEvent<SVGSVGElement>,
  ): [number, number] | null {
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
    setPrototypes((prev) => {
      const next: [number, number][] = [prev[0], prev[1], prev[2]]
      next[dragging] = clamped
      return next
    })
  }

  function snapToClassMeans() {
    const sums: [number, number][] = [
      [0, 0],
      [0, 0],
      [0, 0],
    ]
    const counts = [0, 0, 0]
    for (const s of data) {
      sums[s.y][0] += s.x[0]
      sums[s.y][1] += s.x[1]
      counts[s.y]++
    }
    const means: [number, number][] = [
      [0, 0],
      [0, 0],
      [0, 0],
    ]
    for (let i = 0; i < K; i++) {
      if (counts[i] === 0) continue
      means[i] = [sums[i][0] / counts[i], sums[i][1] / counts[i]]
    }
    setPrototypes(means)
  }

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
        <button
          onClick={snapToClassMeans}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Snap to class means
        </button>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Sharpness ={' '}
            <span className="font-mono">{sharpness.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={6}
            step={0.05}
            value={sharpness}
            onChange={(e) => setSharpness(parseFloat(e.target.value))}
            className="w-36"
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
          {/* Probability heatmap */}
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

          {/* Pairwise boundary lines */}
          {boundaryLines.map((ln, i) => (
            <line
              key={i}
              x1={sx(ln.x1)}
              y1={sy(ln.y1)}
              x2={sx(ln.x2)}
              y2={sy(ln.y2)}
              stroke="var(--ink)"
              strokeWidth={1.2}
              opacity={0.45}
              strokeDasharray="4,3"
            />
          ))}

          {/* Data points */}
          {data.map((s, i) => (
            <circle
              key={i}
              cx={sx(s.x[0])}
              cy={sy(s.x[1])}
              r={4}
              fill={CLASS_HEX[s.y]}
              stroke="var(--paper)"
              strokeWidth={1}
            />
          ))}

          {/* Prototype handles */}
          {prototypes.map((p, i) => (
            <g key={i}>
              <circle
                cx={sx(p[0])}
                cy={sy(p[1])}
                r={14}
                fill={CLASS_HEX[i]}
                opacity={0.15}
              />
              <circle
                cx={sx(p[0])}
                cy={sy(p[1])}
                r={10}
                fill="var(--paper)"
                stroke={CLASS_HEX[i]}
                strokeWidth={2.5}
                className="cursor-grab"
                onPointerDown={(e) => {
                  setDragging(i)
                  e.currentTarget.setPointerCapture(e.pointerId)
                  e.stopPropagation()
                }}
              />
              <text
                x={sx(p[0])}
                y={sy(p[1]) + 4}
                fontSize={11}
                fontFamily="var(--font-mono, monospace)"
                fill={CLASS_HEX[i]}
                textAnchor="middle"
                pointerEvents="none"
              >
                {i}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>K = 3 classes</div>
        <div>
          parameters = {K * 2 + K} ({K}×2 weights + {K} biases)
        </div>
        <div>
          accuracy ={' '}
          <span className="text-ink">{(acc * 100).toFixed(1)}%</span>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 8.2 — Three-class logistic regression on 2D data. Each class
        gets a coloured prototype handle that serves as both its weight
        vector and the centre of its region. Drag the handles to rearrange
        the decision regions; the heatmap blends the three class colours by
        softmax probability, and the dashed lines are the three pairwise
        boundaries meeting at a triple point — a Voronoi-like partition of
        the plane.
      </figcaption>
    </figure>
  )
}
