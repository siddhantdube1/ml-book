'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  makeMoons2,
  makeXOR,
  makeBlobs2,
  buildTreeFrames,
  type Point,
  type TreeNode,
  type TreeFrame,
} from '@/lib/tree'

const PW = 360
const PH = 340
const TW = 380
const TH = 340
const PAD = 22
const T_PAD_X = 30
const T_PAD_T = 26
const T_PAD_B = 24

const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.6
const Y_MAX = 2.6

const CLASS = ['#3c5a8c', '#c7522a', '#5d8a3a'] as const
const CLASS_RGB = [
  [60, 90, 140],
  [199, 82, 42],
  [93, 138, 58],
] as const
const SUB = ['₁', '₂'] as const

const DATASETS = {
  moons: 'Moons',
  xor: 'XOR / checkerboard',
  blobs: 'Three blobs',
} as const
type DatasetKey = keyof typeof DATASETS

function generate(key: DatasetKey): { data: Point[]; numClasses: number } {
  switch (key) {
    case 'moons':
      return { data: makeMoons2(110, 1, 0.16), numClasses: 2 }
    case 'xor':
      return { data: makeXOR(120, 2), numClasses: 2 }
    case 'blobs':
      return { data: makeBlobs2(120, 3, 3), numClasses: 3 }
  }
}

// ── partition cells ──
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

// box of the region reached by a left/right path (for highlighting)
function regionOfPath(root: TreeNode, path: number[]): [number, number, number, number] {
  let x0 = X_MIN, x1 = X_MAX, y0 = Y_MIN, y1 = Y_MAX
  let cur = root
  for (const dir of path) {
    if (cur.kind !== 'split') break
    if (cur.feature === 0) {
      if (dir === 0) x1 = cur.threshold
      else x0 = cur.threshold
    } else {
      if (dir === 0) y1 = cur.threshold
      else y0 = cur.threshold
    }
    cur = dir === 0 ? cur.left : cur.right
  }
  return [x0, x1, y0, y1]
}

// ── tree layout ──
type Laid = { path: string; node: TreeNode; depth: number; slot: number }
function layoutTree(root: TreeNode): { laid: Laid[]; leaves: number; maxDepth: number } {
  const laid: Laid[] = []
  let leaves = 0
  let maxDepth = 0
  const rec = (node: TreeNode, path: string, depth: number): number => {
    maxDepth = Math.max(maxDepth, depth)
    if (node.kind === 'leaf') {
      const slot = leaves++
      laid.push({ path, node, depth, slot })
      return slot
    }
    const l = rec(node.left, path + 'L', depth + 1)
    const r = rec(node.right, path + 'R', depth + 1)
    const slot = (l + r) / 2
    laid.push({ path, node, depth, slot })
    return slot
  }
  rec(root, '', 0)
  return { laid, leaves, maxDepth }
}

export default function TreeGrower() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('moons')
  const [maxDepth, setMaxDepth] = useState(4)
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const { data, numClasses } = useMemo(() => generate(datasetKey), [datasetKey])

  const frames: TreeFrame[] = useMemo(
    () =>
      buildTreeFrames(
        data,
        { numClasses, maxDepth, minSamplesLeaf: 2 },
        12,
      ),
    [data, numClasses, maxDepth],
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
    const t = setTimeout(() => setFrameIdx((i) => i + 1), 650)
    return () => clearTimeout(t)
  }, [playing, frameIdx, frames.length])

  const cur = frames[Math.min(frameIdx, frames.length - 1)]
  const root = cur.root
  const lastStr = cur.lastPath.map((d) => (d === 0 ? 'L' : 'R')).join('')

  const cells = useMemo(() => partitionCells(root, X_MIN, X_MAX, Y_MIN, Y_MAX, []), [root])
  const hi = cur.nSplits > 0 ? regionOfPath(root, cur.lastPath) : null

  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (PW - 2 * PAD)
  const sy = (y: number) => PH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (PH - 2 * PAD)

  // tree layout
  const { laid, leaves, maxDepth: dShown } = useMemo(() => layoutTree(root), [root])
  const posOf = useMemo(() => {
    const m = new Map<string, { x: number; y: number }>()
    const tx = (slot: number) =>
      leaves <= 1
        ? TW / 2
        : T_PAD_X + (slot / (leaves - 1)) * (TW - 2 * T_PAD_X)
    const levelGap = (TH - T_PAD_T - T_PAD_B) / Math.max(dShown, 1)
    const ty = (depth: number) => T_PAD_T + depth * levelGap
    for (const l of laid) m.set(l.path, { x: tx(l.slot), y: ty(l.depth) })
    return m
  }, [laid, leaves, dShown])

  const slotSpacing = leaves <= 1 ? TW : (TW - 2 * T_PAD_X) / Math.max(leaves - 1, 1)
  const boxW = Math.max(16, Math.min(50, slotSpacing * 0.92))
  const showText = boxW > 36

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
              <option key={k} value={k}>{label}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Max depth = <span className="font-mono">{maxDepth}</span></span>
          <input type="range" min={2} max={4} step={1} value={maxDepth} onChange={(e) => setMaxDepth(parseInt(e.target.value))} className="w-28" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <div className="flex flex-col md:flex-row">
          {/* PARTITION PANEL */}
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full md:w-[49%] h-auto block border-b md:border-b-0 md:border-r border-rule">
            {cells.map((c, i) => (
              <rect key={i} x={sx(c.x0)} y={sy(c.y1)} width={sx(c.x1) - sx(c.x0)} height={sy(c.y0) - sy(c.y1)} fill={`rgba(${CLASS_RGB[c.pred].join(',')},0.18)`} stroke="var(--rule)" strokeWidth={0.5} />
            ))}
            {hi && (
              <rect x={sx(hi[0])} y={sy(hi[3])} width={sx(hi[1]) - sx(hi[0])} height={sy(hi[2]) - sy(hi[3])} fill="none" stroke="var(--accent)" strokeWidth={2} />
            )}
            {data.map((p, i) => (
              <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.8} />
            ))}
            <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">feature space</text>
          </svg>

          {/* TREE PANEL */}
          <svg viewBox={`0 0 ${TW} ${TH}`} className="w-full md:w-[51%] h-auto block">
            <text x={T_PAD_X - 8} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">decision tree</text>
            {/* edges */}
            {laid.filter((l) => l.node.kind === 'split').map((l) => {
              const p = posOf.get(l.path)!
              const lc = posOf.get(l.path + 'L')!
              const rc = posOf.get(l.path + 'R')!
              return (
                <g key={`e${l.path}`} stroke="var(--rule)" strokeWidth={1}>
                  <line x1={p.x} y1={p.y} x2={lc.x} y2={lc.y} />
                  <line x1={p.x} y1={p.y} x2={rc.x} y2={rc.y} />
                </g>
              )
            })}
            {/* nodes */}
            {laid.map((l) => {
              const p = posOf.get(l.path)!
              const justSplit = l.path === lastStr && l.node.kind === 'split'
              if (l.node.kind === 'leaf') {
                return (
                  <g key={`n${l.path}`}>
                    <rect x={p.x - 8} y={p.y - 8} width={16} height={16} rx={3} fill={`rgba(${CLASS_RGB[l.node.prediction].join(',')},0.5)`} stroke={CLASS[l.node.prediction]} strokeWidth={1.25} />
                  </g>
                )
              }
              const tStr = l.node.threshold.toFixed(1)
              const cond = `x${SUB[l.node.feature]}<${tStr === '-0.0' ? '0.0' : tStr}`
              return (
                <g key={`n${l.path}`}>
                  <rect x={p.x - boxW / 2} y={p.y - 9} width={boxW} height={18} rx={3} fill="var(--paper)" stroke={justSplit ? 'var(--accent)' : 'var(--rule)'} strokeWidth={justSplit ? 2 : 1} />
                  {showText && (
                    <text
                      x={p.x}
                      y={p.y + 3}
                      fontSize={9}
                      fill="var(--ink)"
                      textAnchor="middle"
                      fontFamily="var(--font-mono, monospace)"
                      style={{ fontFeatureSettings: '"liga" 0, "calt" 0', fontVariantLigatures: 'none' }}
                    >
                      {cond}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        </div>
      </div>

      {/* playback */}
      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button onClick={() => { if (frameIdx >= frames.length - 1) setFrameIdx(0); setPlaying((p) => !p) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[60px]">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setFrameIdx(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <button onClick={() => setFrameIdx((i) => Math.min(i + 1, frames.length - 1))} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Split</button>
        <input type="range" min={0} max={Math.max(frames.length - 1, 1)} value={frameIdx} onChange={(e) => { setPlaying(false); setFrameIdx(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">split {frameIdx} / {frames.length - 1}</span>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>splits so far = <span className="text-ink">{cur.nSplits}</span></div>
        <div>leaves (regions) = <span className="text-ink">{cells.length}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 12.2 — Growing a decision tree, seen two ways at once. Press
        Play and each step makes the single most valuable split: the tree on
        the right gains a node, and the feature space on the left is cut by
        one more axis-aligned line (the new region outlined in teal). On
        moons, the staircase of cuts carves apart two interleaved arcs no
        straight line could separate — and on XOR, a problem hopeless for
        logistic regression, the tree solves it in a handful of splits.
      </figcaption>
    </figure>
  )
}
