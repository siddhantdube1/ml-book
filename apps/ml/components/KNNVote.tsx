'use client'

import { useMemo, useRef, useState } from 'react'
import { makeBlobs2, type Point } from '@/lib/tree'
import { knnNeighbours } from '@/lib/features'

const W = 420
const H = 380
const PAD = 26
const RANGE = 3.2
const CLASS = ['#3c5a8c', '#c7522a'] as const

export default function KNNVote() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [k, setK] = useState(5)
  const [q, setQ] = useState<[number, number]>([0.2, 0.3])
  const [drag, setDrag] = useState(false)

  const data = useMemo<Point[]>(() => makeBlobs2(40, 7, 2), [])

  const sx = (x: number) => PAD + ((x + RANGE) / (2 * RANGE)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + RANGE) / (2 * RANGE)) * (H - 2 * PAD)

  const nn = useMemo(() => knnNeighbours(data, q, k), [data, q, k])
  const votes = useMemo(() => {
    let a = 0, b = 0
    for (const i of nn) (data[i].y === 0 ? (a++) : (b++))
    return { a, b, pred: b > a ? 1 : 0 }
  }, [nn, data])

  const fromPointer = (e: React.PointerEvent<SVGSVGElement>): [number, number] => {
    const svg = svgRef.current!
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    return [Math.max(-RANGE, Math.min(RANGE, ((x - PAD) / (W - 2 * PAD)) * 2 * RANGE - RANGE)), Math.max(-RANGE, Math.min(RANGE, RANGE - ((y - PAD) / (H - 2 * PAD)) * 2 * RANGE))]
  }

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-5 gap-y-2 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2"><span className="text-ink-muted">k = <span className="font-mono">{k}</span></span><input type="range" min={1} max={15} step={1} value={k} onChange={(e) => setK(parseInt(e.target.value))} className="w-44" /></label>
        <span className="font-sans text-xs text-ink-faint">drag the query ◇</span>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block select-none touch-none"
          onPointerMove={(e) => drag && setQ(fromPointer(e))}
          onPointerUp={(e) => { setDrag(false); e.currentTarget.releasePointerCapture(e.pointerId) }}>
          {/* lines to the k neighbours */}
          {nn.map((i) => <line key={`l${i}`} x1={sx(q[0])} y1={sy(q[1])} x2={sx(data[i].x[0])} y2={sy(data[i].x[1])} stroke="var(--ink-muted)" strokeWidth={0.8} opacity={0.5} />)}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={nn.includes(i) ? 6 : 3.4} fill={CLASS[p.y]} stroke={nn.includes(i) ? 'var(--ink)' : 'var(--paper)'} strokeWidth={nn.includes(i) ? 1.6 : 0.6} />
          ))}
          {/* query diamond, coloured by the predicted class */}
          <rect x={sx(q[0]) - 6} y={sy(q[1]) - 6} width={12} height={12} fill={CLASS[votes.pred]} stroke="var(--paper)" strokeWidth={2} transform={`rotate(45 ${sx(q[0])} ${sy(q[1])})`} style={{ cursor: 'grab' }}
            onPointerDown={(e) => { setDrag(true); e.currentTarget.setPointerCapture(e.pointerId) }} />
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>votes: <span style={{ color: CLASS[0] }}>{votes.a} blue</span> · <span style={{ color: CLASS[1] }}>{votes.b} orange</span></div>
        <div>prediction = <span style={{ color: CLASS[votes.pred] }}>{votes.pred === 0 ? 'blue' : 'orange'}</span> (the majority)</div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 4.2 — Predict by asking the neighbours. To classify the query (the
        diamond), kNN finds its k nearest labelled points — ringed, with lines drawn
        to them — and takes a majority vote; the diamond is coloured by the winner.
        Drag it across the plane and the prediction flips as the local majority
        flips. That is the entire algorithm: no training, no equations, just "what
        are the most similar examples, and what were they?"
      </figcaption>
    </figure>
  )
}
