'use client'

import { useMemo, useState } from 'react'
import {
  makeMoons2,
  buildTree,
  accuracy,
  type Point,
  type TreeNode,
} from '@/lib/tree'
import { kFoldSplit } from '@/lib/crossval'

const PW = 360 // partition panel
const PH = 320
const CW = 320 // curve panel
const CH = 320
const PAD = 24
const C_PAD_L = 42
const C_PAD_R = 16
const C_PAD_T = 24
const C_PAD_B = 42

const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.6
const Y_MAX = 2.6
const MAX_D = 10

const CLASS = ['#3c5a8c', '#c7522a'] as const
const CLASS_RGB = [
  [60, 90, 140],
  [199, 82, 42],
] as const

type Cell = { x0: number; x1: number; y0: number; y1: number; pred: number }

function partitionCells(
  node: TreeNode,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  out: Cell[],
): Cell[] {
  if (node.kind === 'leaf') {
    out.push({ x0, x1, y0, y1, pred: node.prediction })
    return out
  }
  if (node.feature === 0) {
    partitionCells(node.left, x0, node.threshold, y0, y1, out)
    partitionCells(node.right, node.threshold, x1, y0, y1, out)
  } else {
    partitionCells(node.left, x0, x1, y0, node.threshold, out)
    partitionCells(node.right, x0, x1, node.threshold, y1, out)
  }
  return out
}

export default function DepthOverfit() {
  const [depth, setDepth] = useState(3)

  const data = useMemo(() => makeMoons2(200, 2, 0.2, 0.08), [])

  // Precompute train + 5-fold CV accuracy for every depth (once).
  const { trainAcc, cvMean, cvStd, bestDepth } = useMemo(() => {
    const folds = kFoldSplit(data.length, 5, 1)
    const trainAcc: number[] = []
    const cvMean: number[] = []
    const cvStd: number[] = []
    for (let d = 1; d <= MAX_D; d++) {
      const opts = { numClasses: 2, maxDepth: d, minSamplesLeaf: 1 }
      trainAcc.push(accuracy(buildTree(data, opts), data))
      const fs = folds.map((f) => {
        const tr = f.trainIdx.map((i) => data[i])
        const te = f.testIdx.map((i) => data[i])
        return accuracy(buildTree(tr, opts), te)
      })
      const m = fs.reduce((a, b) => a + b, 0) / fs.length
      cvMean.push(m)
      cvStd.push(Math.sqrt(fs.reduce((a, b) => a + (b - m) ** 2, 0) / fs.length))
    }
    const bestDepth = cvMean.indexOf(Math.max(...cvMean)) + 1
    return { trainAcc, cvMean, cvStd, bestDepth }
  }, [data])

  const tree = useMemo(
    () => buildTree(data, { numClasses: 2, maxDepth: depth, minSamplesLeaf: 1 }),
    [data, depth],
  )
  const cells = useMemo(
    () => partitionCells(tree, X_MIN, X_MAX, Y_MIN, Y_MAX, []),
    [tree],
  )

  // partition transforms
  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (PW - 2 * PAD)
  const sy = (y: number) => PH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (PH - 2 * PAD)

  // curve transforms
  const cx = (d: number) =>
    C_PAD_L + ((d - 1) / (MAX_D - 1)) * (CW - C_PAD_L - C_PAD_R)
  const cy = (a: number) =>
    CH - C_PAD_B - ((a - 0.5) / 0.5) * (CH - C_PAD_T - C_PAD_B) // 0.5..1.0

  const trainPath = trainAcc
    .map((a, i) => `${i === 0 ? 'M' : 'L'} ${cx(i + 1).toFixed(1)} ${cy(a).toFixed(1)}`)
    .join(' ')
  const cvPath = cvMean
    .map((a, i) => `${i === 0 ? 'M' : 'L'} ${cx(i + 1).toFixed(1)} ${cy(a).toFixed(1)}`)
    .join(' ')

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Max depth = <span className="font-mono">{depth}</span>
          </span>
          <input
            type="range"
            min={1}
            max={MAX_D}
            step={1}
            value={depth}
            onChange={(e) => setDepth(parseInt(e.target.value))}
            className="w-48"
          />
        </label>
        <button
          onClick={() => setDepth(bestDepth)}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Jump to CV-best depth
        </button>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <div className="flex flex-col md:flex-row">
          {/* PARTITION PANEL */}
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full md:w-[53%] h-auto block border-b md:border-b-0 md:border-r border-rule">
            {cells.map((c, i) => (
              <rect
                key={i}
                x={sx(c.x0)}
                y={sy(c.y1)}
                width={sx(c.x1) - sx(c.x0)}
                height={sy(c.y0) - sy(c.y1)}
                fill={`rgba(${CLASS_RGB[c.pred].join(',')},0.18)`}
                stroke="var(--rule)"
                strokeWidth={0.5}
              />
            ))}
            {data.map((p: Point, i) => (
              <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.8} />
            ))}
            <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
              partition at depth {depth} ({cells.length} regions)
            </text>
          </svg>

          {/* CURVE PANEL */}
          <svg viewBox={`0 0 ${CW} ${CH}`} className="w-full md:w-[47%] h-auto block">
            <line x1={C_PAD_L} y1={cy(0.5)} x2={CW - C_PAD_R} y2={cy(0.5)} stroke="var(--rule)" strokeWidth={1} />
            <line x1={C_PAD_L} y1={C_PAD_T} x2={C_PAD_L} y2={cy(0.5)} stroke="var(--rule)" strokeWidth={1} />

            {/* CV-best depth marker */}
            <line x1={cx(bestDepth)} y1={C_PAD_T} x2={cx(bestDepth)} y2={cy(0.5)} stroke="var(--accent)" strokeWidth={1} strokeDasharray="2,3" opacity={0.5} />
            {/* current depth marker */}
            <line x1={cx(depth)} y1={C_PAD_T} x2={cx(depth)} y2={cy(0.5)} stroke="var(--ink)" strokeWidth={1.25} opacity={0.4} />

            {/* training curve */}
            <path d={trainPath} fill="none" stroke="var(--ink-muted)" strokeWidth={1.75} strokeDasharray="4,3" />
            {/* CV error bars + curve */}
            {cvMean.map((a, i) => (
              <line key={i} x1={cx(i + 1)} y1={cy(Math.max(0.5, a - cvStd[i]))} x2={cx(i + 1)} y2={cy(Math.min(1, a + cvStd[i]))} stroke="var(--accent)" strokeWidth={1} opacity={0.4} />
            ))}
            <path d={cvPath} fill="none" stroke="var(--accent)" strokeWidth={2} />
            {cvMean.map((a, i) => (
              <circle key={i} cx={cx(i + 1)} cy={cy(a)} r={i + 1 === bestDepth ? 5 : 3} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1.1} />
            ))}

            {/* ticks */}
            {[0.5, 0.75, 1.0].map((t) => (
              <text key={t} x={C_PAD_L - 6} y={cy(t) + 3} fontSize={10} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{t.toFixed(2)}</text>
            ))}
            {[1, 4, 7, 10].map((d) => (
              <text key={d} x={cx(d)} y={cy(0.5) + 16} fontSize={10} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{d}</text>
            ))}
            <text x={(C_PAD_L + CW - C_PAD_R) / 2} y={CH - 6} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">max depth</text>
            <text x={C_PAD_L} y={C_PAD_T - 8} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">accuracy</text>

            {/* legend */}
            <g transform={`translate(${CW - C_PAD_R - 96}, ${cy(0.5) - 26})`}>
              <line x1={0} y1={0} x2={14} y2={0} stroke="var(--accent)" strokeWidth={2} />
              <text x={18} y={3.5} fontSize={9} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">CV</text>
              <line x1={0} y1={13} x2={14} y2={13} stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" />
              <text x={18} y={16.5} fontSize={9} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">train</text>
            </g>
          </svg>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>train acc = <span className="text-ink">{trainAcc[depth - 1].toFixed(3)}</span></div>
        <div>CV acc = <span className="text-accent">{cvMean[depth - 1].toFixed(3)} ± {cvStd[depth - 1].toFixed(3)}</span></div>
        <div>CV-best depth = {bestDepth}</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 12.3 — A tree overfitting as it deepens. On the left, the
        partition at the chosen depth; on the right, training accuracy (grey,
        dashed) marching to 100% while the cross-validated accuracy (teal,
        with per-fold error bars) peaks at a shallow depth and then falls.
        Past that peak the tree is carving the plane into ever-smaller islands
        to memorise individual noisy points — and cross-validation, the tool
        of Chapter 11, is what tells you where to stop.
      </figcaption>
    </figure>
  )
}
