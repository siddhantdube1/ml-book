'use client'

import { useEffect, useMemo, useState } from 'react'
import { generateDataset, type Point } from '@/lib/datasets'
import { agglomerative, cutTreeK } from '@/lib/clustering'

const W = 680
const H = 380
// left (points) panel
const LX = 20
const LW = 320
// right (dendrogram) panel
const RX = 372
const RW = 296
const TOP = 30
const BOT = 330

const PALETTE = ['#1d6d5e', '#c7522a', '#3c5a8c', '#a06614', '#6b4a8a', '#a83263']
const NEUTRAL = 'var(--ink-faint)'

export default function DendrogramBuilder() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [cutK, setCutK] = useState(3)

  const points = useMemo<Point[]>(() => generateDataset('blobs', 36, 11), [])
  const n = points.length
  const merges = useMemo(() => agglomerative(points, 'average'), [points])

  // build tree: children, heights, members, leaf order, x positions
  const tree = useMemo(() => {
    const total = n + merges.length
    const childA = new Array(total).fill(-1)
    const childB = new Array(total).fill(-1)
    const height = new Array(total).fill(0)
    const members: number[][] = Array.from({ length: total }, (_, i) => (i < n ? [i] : []))
    for (let t = 0; t < merges.length; t++) {
      const node = n + t
      childA[node] = merges[t].a
      childB[node] = merges[t].b
      height[node] = merges[t].height
      members[node] = [...members[merges[t].a], ...members[merges[t].b]]
    }
    const order: number[] = []
    const stack = [total - 1]
    // iterative DFS preserving left-right order
    const dfs = (node: number) => {
      if (node < n) {
        order.push(node)
        return
      }
      dfs(childA[node])
      dfs(childB[node])
    }
    dfs(total - 1)
    const leafX = new Map<number, number>()
    order.forEach((leaf, i) => leafX.set(leaf, i))
    const nodeX = new Array(total).fill(0)
    for (let i = 0; i < n; i++) nodeX[i] = leafX.get(i)!
    for (let t = 0; t < merges.length; t++) {
      const node = n + t
      nodeX[node] = (nodeX[childA[node]] + nodeX[childB[node]]) / 2
    }
    const maxH = Math.max(...merges.map((m) => m.height), 1)
    return { childA, childB, height, members, nodeX, maxH, total }
  }, [merges, n])

  const labels = useMemo(() => cutTreeK(merges, n, cutK), [merges, n, cutK])

  useEffect(() => {
    setStep(0)
    setPlaying(false)
  }, [points])

  useEffect(() => {
    if (!playing) return
    if (step >= merges.length) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setStep((s) => s + 1), 180)
    return () => clearTimeout(t)
  }, [playing, step, merges.length])

  // point-panel mapping (data screen coords -> left box)
  const bounds = useMemo(() => {
    const xs = points.map((p) => p[0])
    const ys = points.map((p) => p[1])
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
  }, [points])
  const px = (x: number) => LX + 16 + ((x - bounds.x0) / (bounds.x1 - bounds.x0)) * (LW - 32)
  const py = (y: number) => TOP + 8 + ((y - bounds.y0) / (bounds.y1 - bounds.y0)) * (BOT - TOP - 16)

  // dendrogram mapping
  const dx = (xi: number) => RX + 8 + (xi / (n - 1)) * (RW - 16)
  const dy = (h: number) => BOT - (h / tree.maxH) * (BOT - TOP)

  const centroid = (mem: number[]): [number, number] => {
    let sx = 0, sy = 0
    for (const i of mem) { sx += points[i][0]; sy += points[i][1] }
    return [sx / mem.length, sy / mem.length]
  }

  // cut height: between the (n-cutK)-th and (n-cutK-1)-th merge heights
  const cutHeight = useMemo(() => {
    const idx = n - cutK
    if (idx <= 0) return tree.maxH * 1.05
    if (idx >= merges.length) return merges[0].height * 0.5
    return (merges[idx - 1].height + merges[idx].height) / 2
  }, [n, cutK, merges, tree.maxH])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Cut into <span className="font-mono">{cutK}</span> clusters</span>
          <input type="range" min={1} max={6} step={1} value={cutK} onChange={(e) => setCutK(parseInt(e.target.value))} className="w-36" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <text x={LX + 4} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">points fusing into clusters</text>
          <text x={RX} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">the dendrogram</text>
          <line x1={356} y1={TOP - 6} x2={356} y2={BOT + 20} stroke="var(--rule)" strokeWidth={1} />

          {/* left: merge links revealed up to step */}
          {merges.slice(0, step).map((m, t) => {
            const ca = centroid(tree.members[m.a])
            const cb = centroid(tree.members[m.b])
            return <line key={`lk${t}`} x1={px(ca[0])} y1={py(ca[1])} x2={px(cb[0])} y2={py(cb[1])} stroke="var(--ink-faint)" strokeWidth={0.8} opacity={0.5} />
          })}
          {points.map((p, i) => (
            <circle key={i} cx={px(p[0])} cy={py(p[1])} r={4} fill={PALETTE[labels[i] % PALETTE.length]} stroke="var(--paper)" strokeWidth={0.7} />
          ))}

          {/* right: dendrogram brackets revealed up to step */}
          {merges.slice(0, step).map((m, t) => {
            const node = n + t
            const yh = dy(m.height)
            const xa = dx(tree.nodeX[m.a])
            const xb = dx(tree.nodeX[m.b])
            const ya = dy(tree.height[m.a])
            const yb = dy(tree.height[m.b])
            const below = m.height < cutHeight
            const col = below ? PALETTE[labels[tree.members[node][0]] % PALETTE.length] : NEUTRAL
            return (
              <g key={`br${t}`} stroke={col} strokeWidth={1.4} fill="none">
                <line x1={xa} y1={ya} x2={xa} y2={yh} />
                <line x1={xb} y1={yb} x2={xb} y2={yh} />
                <line x1={xa} y1={yh} x2={xb} y2={yh} />
              </g>
            )
          })}
          {/* cut line */}
          {step >= merges.length && (
            <>
              <line x1={RX} y1={dy(cutHeight)} x2={RX + RW} y2={dy(cutHeight)} stroke="var(--accent)" strokeWidth={1.25} strokeDasharray="4,3" />
              <text x={RX + RW} y={dy(cutHeight) - 4} fontSize={9} fill="var(--accent)" textAnchor="end" fontFamily="var(--font-mono, monospace)">cut → {cutK}</text>
            </>
          )}
          {/* leaf baseline */}
          <line x1={RX} y1={BOT} x2={RX + RW} y2={BOT} stroke="var(--rule)" strokeWidth={0.75} />
          <text x={RX - 2} y={BOT + 14} fontSize={9} fill="var(--ink-faint)" fontFamily="var(--font-mono, monospace)">height ↑ = merge distance</text>
        </svg>
      </div>

      <div className="flex items-center gap-3 mt-4 font-sans text-sm">
        <button
          onClick={() => { if (step >= merges.length) setStep(0); setPlaying((p) => !p) }}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors min-w-[64px]"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={() => { setStep(0); setPlaying(false) }} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Reset</button>
        <button onClick={() => setStep((s) => Math.min(s + 1, merges.length))} className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors">Step</button>
        <input type="range" min={0} max={merges.length} value={step} onChange={(e) => { setPlaying(false); setStep(parseInt(e.target.value)) }} className="flex-1" />
        <span className="font-mono text-xs text-ink-muted whitespace-nowrap">{step} / {merges.length} merges</span>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 19.1 — Agglomerative clustering. Every point starts alone; at each
        step the two closest clusters merge, drawing a new bracket in the
        dendrogram at the height of the distance they bridged. Play it through
        and the tree builds from the leaves up. The result is not one clustering
        but all of them at once: slide the cut and the dashed line drops through
        the tree, and wherever it severs the branches becomes your clusters —
        the tall vertical gaps are where the data really wants to split.
      </figcaption>
    </figure>
  )
}
