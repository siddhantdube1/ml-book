'use client'

import { useMemo, useState } from 'react'
import {
  makeMoons2,
  makeBlobs2,
  makeXOR,
  buildTree,
  type Point,
  type TreeNode,
} from '@/lib/tree'
import { trainForest, forestVoteProbs, type Forest } from '@/lib/forest'

const PW = 360
const PH = 340
const PAD = 22
const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.6
const Y_MAX = 2.6
const GX = 38
const GY = 28
const MAX_B = 100

const CLASS = ['#3c5a8c', '#c7522a', '#5d8a3a'] as const
const CLASS_RGB = [
  [60, 90, 140],
  [199, 82, 42],
  [93, 138, 58],
]

const DATASETS = {
  moons: 'Moons',
  xor: 'XOR / checkerboard',
  blobs: 'Three blobs',
} as const
type DatasetKey = keyof typeof DATASETS

function generate(key: DatasetKey): { data: Point[]; numClasses: number } {
  switch (key) {
    case 'moons':
      return { data: makeMoons2(150, 1, 0.24), numClasses: 2 }
    case 'xor':
      return { data: makeXOR(160, 2), numClasses: 2 }
    case 'blobs':
      return { data: makeBlobs2(150, 2, 3), numClasses: 3 }
  }
}

function blend(probs: number[]): string {
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
  const K = probs.length
  const certainty = (max - 1 / K) / (1 - 1 / K)
  const alpha = Math.max(0, certainty) * 0.5
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

type Cell = { x0: number; x1: number; y0: number; y1: number; pred: number }
function partitionCells(node: TreeNode, x0: number, x1: number, y0: number, y1: number, out: Cell[]): Cell[] {
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

export default function ForestBoundary() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('moons')
  const [subsample, setSubsample] = useState(true)
  const [b, setB] = useState(60)

  const { data, numClasses } = useMemo(() => generate(datasetKey), [datasetKey])

  const singleTree = useMemo(
    () => buildTree(data, { numClasses, maxDepth: 12, minSamplesLeaf: 1 }),
    [data, numClasses],
  )
  const singleCells = useMemo(
    () => partitionCells(singleTree, X_MIN, X_MAX, Y_MIN, Y_MAX, []),
    [singleTree],
  )

  const forest: Forest = useMemo(
    () =>
      trainForest(data, {
        numTrees: MAX_B,
        numClasses,
        maxDepth: 12,
        minSamplesLeaf: 1,
        maxFeatures: subsample ? 1 : 2,
        seed: 5,
      }),
    [data, numClasses, subsample],
  )

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (PW - 2 * PAD)
  const sy = (y: number) => PH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (PH - 2 * PAD)

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (PW - 2 * PAD) / GX
    const ch = (PH - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = X_MIN + ((i + 0.5) / GX) * (X_MAX - X_MIN)
        const yD = Y_MAX - ((j + 0.5) / GY) * (Y_MAX - Y_MIN)
        const probs = forestVoteProbs(forest, [xD, yD], b)
        cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw + 0.5, h: ch + 0.5, fill: blend(probs) })
      }
    }
    return cells
  }, [forest, b])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Data</span>
          <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value as DatasetKey)} className="bg-paper border border-rule rounded px-2 py-1 text-ink">
            {Object.entries(DATASETS).map(([k, label]) => (
              <option key={k} value={k}>{label}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Trees B = <span className="font-mono">{b}</span></span>
          <input type="range" min={1} max={MAX_B} step={1} value={b} onChange={(e) => setB(parseInt(e.target.value))} className="w-40" />
        </label>
        <label className="flex items-center gap-2 text-ink-muted">
          <input type="checkbox" checked={subsample} onChange={(e) => setSubsample(e.target.checked)} />
          Feature subsampling
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <div className="flex flex-col md:flex-row">
          {/* SINGLE TREE */}
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full md:w-[50%] h-auto block border-b md:border-b-0 md:border-r border-rule">
            {singleCells.map((c, i) => (
              <rect key={i} x={sx(c.x0)} y={sy(c.y1)} width={sx(c.x1) - sx(c.x0)} height={sy(c.y0) - sy(c.y1)} fill={`rgba(${CLASS_RGB[c.pred].join(',')},0.2)`} stroke="var(--rule)" strokeWidth={0.4} />
            ))}
            {data.map((p, i) => (
              <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={2.6} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />
            ))}
            <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">one tree (blocky)</text>
          </svg>

          {/* FOREST VOTE MAP */}
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full md:w-[50%] h-auto block">
            {heatmap.map((c, i) => (
              <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} />
            ))}
            {data.map((p, i) => (
              <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={2.6} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />
            ))}
            <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">forest of {b} (smooth vote)</text>
          </svg>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 13.2 — The random forest, made visible. On the left, a single
        fully-grown tree: its boundary is blocky and axis-aligned, and it
        clings to individual points. On the right, the forest's vote map —
        at each location, the fraction of the B trees voting for each class,
        blended to a colour. Slide B up and the staircases of a hundred
        different trees average into a smooth, soft, stable boundary; the
        graded colour near the frontier is genuine uncertainty. Toggle
        feature subsampling to see the trees decorrelate and the map settle
        faster.
      </figcaption>
    </figure>
  )
}
