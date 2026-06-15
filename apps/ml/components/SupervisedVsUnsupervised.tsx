'use client'

import { useMemo } from 'react'
import { makeBlobs2, type Point } from '@/lib/tree'
import { runKMeans } from '@/lib/kmeans'

const W = 680
const H = 340
const CLASS = ['#3c5a8c', '#c7522a', '#5d8a3a'] as const
const GROUP = ['#1d6d5e', '#a06614', '#6b4a8a'] as const

export default function SupervisedVsUnsupervised() {
  const data = useMemo<Point[]>(() => makeBlobs2(150, 11, 3), [])

  // unsupervised: discover groups with k-means (no labels used)
  const clusters = useMemo(() => {
    const pts = data.map((p) => p.x as [number, number])
    const frames = runKMeans(pts, 3, 'kpp', 2)
    return frames[frames.length - 1].assignments
  }, [data])

  const bounds = useMemo(() => {
    const xs = data.map((p) => p.x[0])
    const ys = data.map((p) => p.x[1])
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
  }, [data])

  const PW = 320
  const PH = 250
  const px = (x: number, ox: number) => ox + 16 + ((x - bounds.x0) / (bounds.x1 - bounds.x0)) * (PW - 32)
  const py = (y: number, oy: number) => oy + PH - 22 - ((y - bounds.y0) / (bounds.y1 - bounds.y0)) * (PH - 40)

  return (
    <figure className="my-10">
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <text x={6 + PW / 2} y={24} fontSize={12} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">supervised — labels are given</text>
          <text x={354 + PW / 2} y={24} fontSize={12} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">unsupervised — no labels</text>

          {/* supervised: coloured by the labels you provide */}
          {data.map((p, i) => (
            <circle key={`s${i}`} cx={px(p.x[0], 6)} cy={py(p.x[1], 40)} r={3.4} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />
          ))}
          <text x={6 + PW / 2} y={H - 8} fontSize={10} fill="var(--ink-faint)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">you give the answers → it learns to predict them</text>

          <line x1={340} y1={34} x2={340} y2={H - 18} stroke="var(--rule)" strokeWidth={1} />

          {/* unsupervised: same points, groups discovered by structure alone */}
          {data.map((p, i) => (
            <circle key={`u${i}`} cx={px(p.x[0], 354)} cy={py(p.x[1], 40)} r={3.4} fill="none" stroke={GROUP[clusters[i]]} strokeWidth={1.8} />
          ))}
          <text x={354 + PW / 2} y={H - 8} fontSize={10} fill="var(--ink-faint)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">no answers → it finds the groups itself</text>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 1.2 — Two ways to learn from the same data. On the left, every point
        comes with a label (its colour), and the task is <em>supervised</em>: learn
        to predict the label of a new point — the setting of most of this book. On
        the right, the points arrive with no labels at all, and the task is{' '}
        <em>unsupervised</em>: with nothing but the positions to go on, the model
        discovers the groups hiding in the data. Same points, two utterly
        different questions — and the second one needs no answer key at all.
      </figcaption>
    </figure>
  )
}
