'use client'

import { useMemo, useState } from 'react'
import { generateDataset, type DatasetShape, type Point } from '@/lib/datasets'
import { agglomerative, cutTreeK, dbscan } from '@/lib/clustering'
import { runKMeans } from '@/lib/kmeans'

const W = 680
const H = 320
const PALETTE = ['#1d6d5e', '#c7522a', '#3c5a8c', '#a06614', '#6b4a8a', '#a83263']

const SHAPES: { key: DatasetShape; label: string }[] = [
  { key: 'blobs', label: 'blobs' },
  { key: 'moons', label: 'moons' },
  { key: 'circles', label: 'circles' },
  { key: 'aniso', label: 'aniso' },
]
const CFG: Record<string, { k: number; eps: number }> = {
  blobs: { k: 3, eps: 36 },
  moons: { k: 2, eps: 22 },
  circles: { k: 2, eps: 46 },
  aniso: { k: 3, eps: 30 },
}
const MINPTS = 4

export default function ShapeShowdown() {
  const [shape, setShape] = useState<DatasetShape>('moons')
  const points = useMemo<Point[]>(() => generateDataset(shape, 160, 5), [shape])
  const cfg = CFG[shape]

  const kmeans = useMemo(() => {
    const f = runKMeans(points, cfg.k, 'kpp', 1)
    return f[f.length - 1].assignments
  }, [points, cfg.k])
  const hier = useMemo(() => cutTreeK(agglomerative(points, 'single'), points.length, cfg.k), [points, cfg.k])
  const db = useMemo(() => dbscan(points, cfg.eps, MINPTS), [points, cfg.eps])

  const bounds = useMemo(() => {
    const xs = points.map((p) => p[0])
    const ys = points.map((p) => p[1])
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
  }, [points])

  const PW = 222
  const PH = 250
  const offsets = [4, 230, 456]
  const px = (x: number, ox: number) => ox + 12 + ((x - bounds.x0) / (bounds.x1 - bounds.x0)) * (PW - 24)
  const py = (y: number, oy: number) => oy + 28 + ((y - bounds.y0) / (bounds.y1 - bounds.y0)) * (PH - 44)

  const panels = [
    { title: 'k-means', labels: kmeans },
    { title: 'hierarchical (single)', labels: hier },
    { title: `DBSCAN (ε=${cfg.eps})`, labels: db.labels },
  ]

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-2 mb-4 font-sans text-sm">
        {SHAPES.map((s) => (
          <button key={s.key} onClick={() => setShape(s.key)} className="px-3 py-1 border rounded transition-colors" style={{ borderColor: shape === s.key ? 'var(--accent)' : 'var(--rule)', background: shape === s.key ? 'var(--accent)' : 'transparent', color: shape === s.key ? 'var(--paper)' : 'var(--ink-muted)' }}>
            {s.label}
          </button>
        ))}
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {panels.map((panel, pi) => {
            const ox = offsets[pi]
            return (
              <g key={panel.title}>
                <text x={ox + PW / 2} y={18} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-sans, sans-serif)">{panel.title}</text>
                {points.map((p, i) => {
                  const lab = panel.labels[i]
                  if (lab < 0) {
                    const x = px(p[0], ox)
                    const y = py(p[1], 30)
                    return (
                      <g key={i} stroke="var(--ink-faint)" strokeWidth={1}>
                        <line x1={x - 2.5} y1={y - 2.5} x2={x + 2.5} y2={y + 2.5} />
                        <line x1={x - 2.5} y1={y + 2.5} x2={x + 2.5} y2={y - 2.5} />
                      </g>
                    )
                  }
                  return <circle key={i} cx={px(p[0], ox)} cy={py(p[1], 30)} r={2.8} fill={PALETTE[lab % PALETTE.length]} stroke="var(--paper)" strokeWidth={0.4} />
                })}
              </g>
            )
          })}
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 19.4 — Why not just use k-means? On round <strong>blobs</strong>
        all three agree. But switch to <strong>moons</strong> or{' '}
        <strong>circles</strong> and k-means&rsquo; straight-line partition cuts
        clean through the shapes, pairing the wrong halves — while
        single-linkage hierarchical and DBSCAN trace the true forms, DBSCAN also
        flagging the stray points (grey ×) as noise. Each algorithm answers a
        different question; only some of those questions fit your data.
      </figcaption>
    </figure>
  )
}
