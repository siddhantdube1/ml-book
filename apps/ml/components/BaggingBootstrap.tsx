'use client'

import { useMemo, useState } from 'react'
import { makeMoons2, buildTree, type Point, type TreeNode } from '@/lib/tree'
import { bootstrap } from '@/lib/forest'

const PW = 360
const PH = 320
const PAD = 22
const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.6
const Y_MAX = 2.6

const CLASS = ['#3c5a8c', '#c7522a'] as const
const CLASS_RGB = [
  [60, 90, 140],
  [199, 82, 42],
] as const

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

export default function BaggingBootstrap() {
  const [seed, setSeed] = useState(1)

  const data = useMemo<Point[]>(() => makeMoons2(80, 4, 0.2), [])

  const bs = useMemo(() => bootstrap(data.length, seed), [data, seed])
  const tree = useMemo(() => {
    const sample = bs.idx.map((i) => data[i])
    return buildTree(sample, { numClasses: 2, maxDepth: 6, minSamplesLeaf: 1 })
  }, [bs, data])
  const cells = useMemo(() => partitionCells(tree, X_MIN, X_MAX, Y_MIN, Y_MAX, []), [tree])

  const oobFrac = bs.oob.length / data.length

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (PW - 2 * PAD)
  const sy = (y: number) => PH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (PH - 2 * PAD)

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          New bootstrap
        </button>
        <span className="text-ink-muted">
          out-of-bag this draw ={' '}
          <span className="font-mono text-ink">{(oobFrac * 100).toFixed(0)}%</span>
        </span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <div className="flex flex-col md:flex-row">
          {/* BOOTSTRAP PANEL */}
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full md:w-[50%] h-auto block border-b md:border-b-0 md:border-r border-rule">
            <rect x={PAD} y={PAD} width={PW - 2 * PAD} height={PH - 2 * PAD} fill="none" stroke="var(--rule)" strokeWidth={1} />
            {data.map((p, i) => {
              const c = bs.counts[i]
              const oob = c === 0
              const r = oob ? 3 : 3 + Math.min(c - 1, 3) * 1.4
              return (
                <circle
                  key={i}
                  cx={sx(p.x[0])}
                  cy={sy(p.x[1])}
                  r={r}
                  fill={oob ? 'none' : CLASS[p.y]}
                  stroke={oob ? 'var(--ink-faint)' : 'var(--paper)'}
                  strokeWidth={oob ? 1.25 : 0.8}
                  opacity={oob ? 0.8 : 1}
                />
              )
            })}
            <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
              the bootstrap sample
            </text>
            <text x={PAD} y={PH - PAD + 16} fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
              filled = drawn (bigger = drawn more); hollow = out-of-bag
            </text>
          </svg>

          {/* TREE PARTITION PANEL */}
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full md:w-[50%] h-auto block">
            {cells.map((c, i) => (
              <rect key={i} x={sx(c.x0)} y={sy(c.y1)} width={sx(c.x1) - sx(c.x0)} height={sy(c.y0) - sy(c.y1)} fill={`rgba(${CLASS_RGB[c.pred].join(',')},0.18)`} stroke="var(--rule)" strokeWidth={0.5} />
            ))}
            {data.map((p, i) => (
              <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={2.5} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} opacity={bs.counts[i] === 0 ? 0.3 : 1} />
            ))}
            <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
              the tree it grows
            </text>
          </svg>
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 13.1 — Bagging starts here. A bootstrap sample draws as many
        points as the dataset has, but with replacement: some points are
        picked several times (drawn larger), and about a third are never
        picked at all (hollow — the out-of-bag set). The tree on the right is
        grown on that resample. Press "New bootstrap" and watch both the
        sample and the tree it grows change — that built-in instability is
        exactly the diversity a forest will average over.
      </figcaption>
    </figure>
  )
}
